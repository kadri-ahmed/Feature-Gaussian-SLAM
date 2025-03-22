import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from clustering.association_visualizer import plot_cluster_language_association, plot_single_cluster
from splatam_utils.slam_helpers import (
    transformed_params2depthplussilhouette,
    transform_to_frame
)
from line_profiler import *
# from memory_profiler import *

def calculate_iou(masks1, masks2, base=None, batch_size=8):
    """
    Intersection over Union (IoU) between two sets of masks,
    batching only over masks1.
    Args:
        masks1: PyTorch tensor of shape [n, H, W], dtype=torch.bool.
        masks2: PyTorch tensor of shape [m, H, W], dtype=torch.bool.
        base: Determines the union calculation ("former", "later", or None for default IoU).
        batch_size: Number of masks from masks1 to process in one batch.
    Returns:
        iou_matrix: PyTorch tensor of shape [m, n], containing IoU values.
    """
    # Ensure the masks are boolean
    if masks1.dtype != torch.bool:
        masks1 = masks1.to(torch.bool)
    if masks2.dtype != torch.bool:
        masks2 = masks2.to(torch.bool)

    m, h, w = masks2.shape
    n, _, _ = masks1.shape

    # Initialize the IoU matrix
    iou_matrix = torch.zeros((m, n), device=masks1.device, dtype=torch.float32)

    # Expand masks2 once to avoid repeated expansion
    masks2_expanded = masks2.unsqueeze(1)  # [m, 1, H, W]

    # Process masks1 in batches
    for j in range(0, n, batch_size):
        masks1_batch = masks1[j:j+batch_size]  # Take a batch of masks1 [batch_size, H, W]
        masks1_batch_expanded = masks1_batch.unsqueeze(0)  # [1, batch_size, H, W]

        # Compute intersection
        intersection = (masks2_expanded & masks1_batch_expanded).float().sum(dim=(2, 3))  # [m, batch_size]

        # Compute union
        if base == "former":
            union = masks1_batch_expanded.float().sum(dim=(2, 3)).transpose(0, 1) + 1e-6  # [batch_size, 1]
        elif base == "later":
            union = masks2_expanded.float().sum(dim=(2, 3)) + 1e-6  # [m, 1]
        else:
            union = (masks2_expanded | masks1_batch_expanded).float().sum(dim=(2, 3)) + 1e-6  # [m, batch_size]

        # Compute IoU for this batch
        iou_matrix[:, j:j+batch_size] = intersection / union
        
    return iou_matrix

def get_transformed_gaussians(params, tracking, mapping, do_ba, iter_time_idx):
    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)
        
    return transformed_gaussians

def calculate_iou_single(mask1, mask2, base=None):
    if mask1.dtype != torch.bool:
        mask1 = mask1.to(torch.bool)
    if mask2.dtype != torch.bool:
        mask2 = mask2.to(torch.bool)
    
    intersection = (mask1 & mask2).sum().float()
    
    if base == "former":
        union = mask1.sum().float() + 1e-6
    elif base == "later":
        union = mask2.sum().float() + 1e-6
    else:
        union = (mask1 | mask2).sum().float() + 1e-6
    
    return intersection / union

class GaussiansClustering:
    def __init__(self):
        self.num_clusters = 0
        self.cluster_id = 0
        self.assignments = torch.empty(0, dtype=torch.bool, device="cuda:0")
        self.confidences = torch.empty(0)
        self.cluster_masks = {}
        self.cam_data_collection = {}
        self.cluster_features = {}
    
    def render_sil(self, params, precomp_render_var, curr_data, iter_time_idx, gaussian_indices, cluster_index, rendered_cluster_features, device="cuda:0"):            
        filtered_gaussians_idx = gaussian_indices#torch.nonzero(cluster_indices == cluster_idx, as_tuple=True)[0]

        # selected_params = {'means3D': params['means3D'][filtered_gaussians_idx],
        #                 #'means2D': params['means2D'][filtered_gaussians_idx],
        #                 'rgb_colors': params['rgb_colors'][filtered_gaussians_idx, 3:6],
        #                     'unnorm_rotations': params['unnorm_rotations'][filtered_gaussians_idx],
        #                     'logit_opacities': params['logit_opacities'][filtered_gaussians_idx],
        #                     'log_scales': params['log_scales'][filtered_gaussians_idx],
        #                     # 'semantic_features': params['semantic_features'][filtered_gaussians_idx],
        #                     'cam_unnorm_rots': params['cam_unnorm_rots'],
        #                     'cam_trans': params['cam_trans']}
        
        # render cluster
        with torch.no_grad():
            #transformed_gaussians = get_transformed_gaussians(params, False, True, False, iter_time_idx)
            # selected_transformed_gaussians = {}
            # for key, item in transformed_gaussians.items():
            #     selected_transformed_gaussians[key] = item[filtered_gaussians_idx]
            
            selected_render_var = {}
            
            for key, item in precomp_render_var.items():
                selected_render_var[key] = item[filtered_gaussians_idx]
                    
            # rendervar_sil = transformed_params2depthplussilhouette(selected_params, curr_data['w2c'],
            #                                                     selected_transformed_gaussians)

            # rend_sil, _, _, _, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar_sil)
            rend_sil, _, _, _, _ = Renderer(raster_settings=curr_data['cam'])(**selected_render_var)
       
            rendered_cluster_features[cluster_index] = rend_sil.detach().cpu()


    def add_new_cluster_column(self, assignments, device="cuda:0"):
        num_gaussians = assignments.size(0)
        new_col = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
        if assignments.numel() == 0:
            assignments = new_col
        else:
            assignments = torch.cat([assignments, new_col], dim=1)

    def get_cluster_language_features(self, mask_language_features):
        
        cluster_lang_features = {}
        
        for cluster_idx, all_mask_infos in self.cluster_masks.items():

            all_features_curr_cluster = torch.zeros((len(all_mask_infos), 512))
            
            for i, (mask_time_idx, mask_index) in enumerate(all_mask_infos):
                all_features_curr_cluster[i] = mask_language_features[mask_time_idx][mask_index]
                
            similarity_matrix = F.cosine_similarity(all_features_curr_cluster.unsqueeze(1), all_features_curr_cluster.unsqueeze(0), dim=-1)
            similarity_matrix.fill_diagonal_(0)
            mean_similarities = similarity_matrix.mean(dim=1)
            
            best_index = mean_similarities.argmax()
            
            cluster_lang_features[cluster_idx] = all_features_curr_cluster[best_index]
            
        return cluster_lang_features

    # def update_cluster_features(self):
        

    # @profile
    def update_gaussian_clusters(self, params, curr_data, variables, iter_time_idx, gaussian_contr, cluster_input, mask_language_features, rendered_cluster_features, device="cuda:0"):
        # render mask gaussians
        # keep track of current frame masks, to not compare masks of current frame with each other
        # filter out sam masks/clusters from all clusters before with dissimilar clip to current sam/cluster
        # calculate IoU with sam masks or union(sam masks , rendered cluster), between current and before clusters
        # if even small IoU we can merge
        # output:
        # assignments [num_gaussians, num_clusters]
        # 
        
        # first get all gaussians that are visible in the current frame
        transformed_gaussians = get_transformed_gaussians(params, False, True, False, iter_time_idx)
        rendervar_sil = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                transformed_gaussians)
                
        # safe camera data
        # self.cam_data_collection[iter_time_idx] = curr_data
        
        responsible_gaussians_per_pixel = gaussian_contr[0]
        sam_masks = cluster_input['sam_masks'][iter_time_idx]
        num_gaussians = params['means3D'].size(0)
        num_masks = sam_masks.size(0)
        current_assignments = torch.zeros(num_gaussians, num_masks, dtype=torch.bool, device=device)
        
        if num_masks == 0:
            return

            
        # transformed_params_for_rendering = transform_to_frame(params, iter_time_idx, False, False)
        
        
        current_frame_lang_features = torch.zeros((num_masks, 512))
        # render current clusters sils
        for mask_id in range(num_masks):
            new_cluster_index = self.num_clusters + mask_id#self.cluster_id
            # participating_gaussians = responsible_gaussians_per_pixel * sam_masks[mask_id].cuda()
            # participating_gaussians = torch.unique(participating_gaussians)
            participating_gaussians = torch.unique(responsible_gaussians_per_pixel[0][sam_masks[mask_id].cuda()])
            participating_gaussians = participating_gaussians[participating_gaussians != -1]
            
            current_assignments[participating_gaussians.long(), mask_id] = True

            # cluster_language_features[new_cluster_index] = mask_language_features[iter_time_idx][mask_id]
            current_frame_lang_features[mask_id] = mask_language_features[iter_time_idx][mask_id]

            # self.render_sil(params, curr_data, iter_time_idx,
            # self.render_sil(params, transformed_params_for_rendering, curr_data, iter_time_idx, 
            #                 torch.where(current_assignments[:, mask_id] == True)[0], 
            #                 new_cluster_index, rendered_cluster_features)
            self.render_sil(params, rendervar_sil, curr_data, iter_time_idx, 
                            torch.where(current_assignments[:, mask_id] == True)[0], 
                            new_cluster_index, rendered_cluster_features)
            #self.cluster_id += 1

        if self.num_clusters == 0:#torch.all(self.assignments == 0):
            self.assignments = current_assignments
            self.num_clusters = self.assignments.size(1)
            
            # save the mask per cluster (initial), for every cluster, we save the time_idx and the mask_idx at that frame
            for mask_id in range(num_masks):
                self.cluster_masks[mask_id] = [(iter_time_idx, mask_id)] 
                # self.cluster_features = mask_language_features[mask_id]
            
            debug_assignments(self.assignments)
            
            # update_cluster_features()
            
            self.assignments = self.assignments.to('cpu')
            return
        else:
            # extend tensor by number of new gaussians
            self.assignments = self.assignments.to('cuda')
            self.assignments = torch.cat((self.assignments, torch.full((num_gaussians - self.assignments.size(0), self.assignments.size(1)), False, dtype=torch.bool).cuda()))
            debug_assignments(self.assignments)

        # render old cluster sils
        # but only consider visible clusters
        
        # visible_gaussians_mask = Renderer(raster_settings=curr_data['cam']).ma
        
        visible_gaussians_mask = Renderer(raster_settings=curr_data['cam']).markVisible(rendervar_sil['means3D'])
        
        if visible_gaussians_mask.sum() != num_gaussians:
            print(f"Gaussians not visible: {num_gaussians - visible_gaussians_mask.sum()}")
        
        # # we can use this mask to check which clusters are visible
        visible_clusters = (self.assignments & visible_gaussians_mask.unsqueeze(1)).any(dim=0).nonzero().squeeze()
        
        if visible_clusters.size(0) != self.num_clusters:
            print(f"Clusters not visible: {self.num_clusters - visible_clusters.size(0)}")
        
        if visible_clusters.dim() > 0:
        
            for visible_cluster_index in visible_clusters:# in range(self.assignments.size(1)):
                #column = self.assignments[:, i]
                #cluster_id = torch.unique(column[column != -1]) 
                # self.render_sil(params, curr_data, iter_time_idx,
                # self.render_sil(params, transformed_params_for_rendering, curr_data, iter_time_idx, 
                #                 torch.where(self.assignments[:,i] == True)[0], 
                #                 i, rendered_cluster_features)
                self.render_sil(params, rendervar_sil, curr_data, iter_time_idx, 
                                torch.where(self.assignments[:,visible_cluster_index] == True)[0], 
                                visible_cluster_index.item(), rendered_cluster_features)

            # merge clusters
            # clip similarity > 0.9
            # visible clusters contains the indices/IDs of all clusters visible in this frame
            old_cluster_ind = visible_clusters#torch.arange(0, self.assignments.size(1))
            current_cluster_features_compute = self.get_cluster_language_features(mask_language_features)
            old_lang_features = torch.stack([current_cluster_features_compute[feat.item()] for feat in old_cluster_ind])
            # old_lang_features = torch.stack([mask_language_features[self.cluster_masks[ind.item()][-1][0]][self.cluster_masks[ind.item()][-1][1]] for ind in old_cluster_ind])
    
                
        new_cluster_ind = torch.arange(self.assignments.size(1), self.assignments.size(1) + num_masks)
        #old_lang_features = torch.stack([mask_language_features[v[-1][0]][v[-1][1]] for k, v in self.cluster_masks.items() if k in old_cluster_ind])
        new_lang_features = current_frame_lang_features
        # old_lang_features = torch.stack([cluster_language_features[old_ind] for old_ind in old_cluster_ind.tolist()])#cluster_language_features[old_cluster_ind]#get_clips_for_cluster(cluster_language_features, self.assignments)
        # new_lang_features = torch.stack([cluster_language_features[new_ind] for new_ind in new_cluster_ind.tolist()])#cluster_language_features[new_cluster_ind]#get_clips_for_cluster(cluster_language_features, current_assignments)

        if visible_clusters.dim() == 0:
            for i, new_cid in enumerate(new_cluster_ind):
                new_cid = new_cid.item()
                
                self.assignments = torch.cat([self.assignments, current_assignments[:, i].unsqueeze(1)], dim=1)
                # real_new_cluster_inds.append(self.num_clusters)
                rendered_cluster_features[self.num_clusters] = rendered_cluster_features.pop(new_cid)
                # rendered_cluster_features.remove
                # save masks
                assert self.num_clusters not in self.cluster_masks.keys()
                self.cluster_masks[self.num_clusters] = [(iter_time_idx, i)]
                self.num_clusters += 1
            
            self.assignments = self.assignments.to('cpu')
            return

        if old_lang_features.size(0) == 0 or new_lang_features.size(0) == 0:  
            return
        
        # old_features = torch.stack(old_features_list).cuda()  # shape: (N_old, D)
        # new_features = torch.stack(new_features_list).cuda()  # shape: (N_new, D)

        if iter_time_idx >= 0:
    
            for cluster in visible_clusters:
                plot_single_cluster(rendered_cluster_features[cluster.item()], curr_data['im'], str(cluster.item()), "debug_vis/old_cluster_")
            # for i, cluster in enumerate(list(rendered_cluster_features.values())[:self.num_clusters]):
            #     plot_single_cluster(cluster, curr_data['im'], str(i), "debug_vis/old_cluster_")
        
            for i, cluster in enumerate(list(rendered_cluster_features.values())[self.num_clusters:]):
                plot_single_cluster(cluster, curr_data['im'], str(i), "debug_vis/new_cluster_")


        similarity_matrix = F.cosine_similarity(
            new_lang_features.unsqueeze(1).cuda(),  # -> shape: (N_new, 1, D)
            old_lang_features.unsqueeze(0).cuda(),  # -> shape: (1, N_old_visible, D)
            dim=-1
        )
        
        all_rendered_clusters_tensor = torch.stack([v for k, v in rendered_cluster_features.items() if k in visible_clusters], dim=0)
        all_new_clusters =  torch.stack([v for k, v in rendered_cluster_features.items() if k in new_cluster_ind], dim=0)
        all_ious = calculate_iou(all_rendered_clusters_tensor.sum(dim=1), all_new_clusters.sum(dim=1))

        summes_scores = 2 * similarity_matrix + all_ious.cuda()
        
        max_score_per_new_mask, max_ind_per_new_mask = summes_scores.max(dim=1)
        
        real_new_cluster_inds = []
        for i, new_cid in enumerate(new_cluster_ind):
            new_cid = new_cid.item()
            
            current_score = max_score_per_new_mask[i]
            
            index_incoming = i
            index_existing = max_ind_per_new_mask[i]  
            
            index_existing_cluster = visible_clusters[index_existing]          
            
            if current_score > 2.0:
                bool_old = all_rendered_clusters_tensor[index_existing].sum(dim=0).bool()
                bool_new = all_new_clusters[index_incoming].sum(dim=0).bool()
                participating_old = torch.unique(responsible_gaussians_per_pixel[0][(bool_old.cuda() & ~bool_new.cuda())])#torch.unique(participating_old)
                self.assignments[participating_old.long(),index_existing_cluster] = False
                self.assignments[:,index_existing_cluster] = self.assignments[:,index_existing_cluster] | current_assignments[:,i]
                del rendered_cluster_features[new_cid]
                    
                # add mask
                assert index_existing_cluster.item() in self.cluster_masks.keys()
                self.cluster_masks[index_existing_cluster.item()].append((iter_time_idx, i))
            else:
                self.assignments = torch.cat([self.assignments, current_assignments[:, i].unsqueeze(1)], dim=1)
                real_new_cluster_inds.append(self.num_clusters)
                rendered_cluster_features[self.num_clusters] = rendered_cluster_features.pop(new_cid)
                # save masks
                self.cluster_masks[self.num_clusters] = [(iter_time_idx, i)]
                self.num_clusters += 1
        
        # TODO: determine how many
        # take_top_k = min(visible_clusters.size(0), 5)
        # best_three_scores, best_three_matches = torch.topk(similarity_matrix, take_top_k, dim=1)
        
       
        # # all_ious = all_ious.fill_diagonal_(0)

        # # calculate IoU 
        # for i, new_cid in enumerate(new_cluster_ind):
        #     new_cid = new_cid.item()
            
        #     old_cids = old_cluster_ind[best_three_matches[i]]
        #     scores = best_three_scores[i]
        #     #new_cluster_areas = cluster_input['sam_masks'][iter_time_idx][i]
        #     new_cluster_areas = rendered_cluster_features[new_cid]
        #     old_cluster_areas =  torch.stack([rendered_cluster_features[visible_clusters[best_k_ind].item()] for best_k_ind in best_three_matches[i]])#torch.stack([r for k,r in rendered_cluster_features.items() if k in old_cids.tolist()], dim=0)
        #     #old_cluster_areas = torch.stack([r for k,r in rendered_cluster_features.items() if k in old_cids.tolist()], dim=0)
        #     #old_cluster_areas = {k :r for k, r in rendered_cluster_features.items() if k in old_cids.tolist()}#rendered_cluster_features[old_cids]
        #     ious = calculate_iou_new(new_cluster_areas.sum(dim=0)[None,:], old_cluster_areas.sum(dim=1)).squeeze()
        #     #ious = calculate_iou_new(new_cluster_areas[None,:], old_cluster_areas.sum(dim=1)).squeeze()
            
        #     found_match = False
        #     for best_k in range(take_top_k):
        #         current_score = scores[best_k]
        #         current_iou = ious[best_k]
        #         current_old_cid = old_cids[best_k]
        #         if (current_score > 0.9 and current_iou > 0.15) or (current_iou > 0.7 and current_score > 0.7) or (current_iou + current_score > 1.4) or (current_score > 0.99 and current_iou > 0.0):
        #             found_match = True
                    
        #             # now update the assignments based on the current frame
        #             # use sam mask as ground truth -> gaussians that are rendered in pixels that the sam_mask
        #             # does not include, get 'declustered'
                    
        #             # gaussians_to_uncluster = old_cluster_areas[best_k] & sam_masks[i]
        #             bool_old = old_cluster_areas[best_k].sum(dim=0).bool()
        #             bool_new = new_cluster_areas.sum(dim=0).bool()
        #             # participating_old = responsible_gaussians_per_pixel * (bool_old.cuda() & ~bool_new.cuda())
        #             participating_old = torch.unique(responsible_gaussians_per_pixel[0][(bool_old.cuda() & ~bool_new.cuda())])#torch.unique(participating_old)
        #             # participating_old = participating_old[participating_old >= 0]
        #             #num_oold = participating_old.size(0)
        #             #participating_old = participating_old[visible_gaussians_mask[participating_old.long()]]
        #             #if (num_oold - participating_old.size(0)) != 0:
        #             #    print("alarm")
        #             nuber_before = self.assignments[:, current_old_cid].sum()
        #             self.assignments[participating_old.long(),current_old_cid] = False
        #             # if nuber_before != self.assignments[:, current_old_cid].sum():
        #             #     print("filtering successfull")
        #             self.assignments[:,current_old_cid] = self.assignments[:,current_old_cid] | current_assignments[:,i]
                    
        #             # col_index = (self.assignments == current_old_cid).any(dim=0).nonzero(as_tuple=True)[0]
        #             # curr_indices = torch.nonzero(current_assignments[:, i] != -1, as_tuple=True)[0].unique().cuda()
        #             # old_indices = torch.nonzero(self.assignments[:, col_index] != -1, as_tuple=True)[0].unique().cuda()
        #             # merged_col_indices = torch.unique(torch.cat([curr_indices, old_indices]))
        #             # self.assignments[merged_col_indices, col_index] = current_old_cid        # merge
                    
        #             #rendered_cluster_features
        #             del rendered_cluster_features[new_cid]
        #             #self.num_clusters -= 1
                    
        #             # add mask
        #             assert current_old_cid.item() in self.cluster_masks.keys()
        #             self.cluster_masks[current_old_cid.item()].append((iter_time_idx, i))
                    
        #             break
                
        #     if found_match == False:
                
        #         # for jjj, cluster in enumerate(list(rendered_cluster_features.values())[:self.num_clusters]):
        #         #     plot_single_cluster(cluster, curr_data['im'], str(jjj), "debug_vis/old_cluster_")
            
        #         # for jjj, cluster in enumerate(list(rendered_cluster_features.values())[self.num_clusters:]):
        #         #     plot_single_cluster(cluster, curr_data['im'], str(jjj), "debug_vis/new_cluster_")

        #         max_iou, max_ind = all_ious[i].max(dim=0)
        #         if max_iou > 0.75:
                    
                
        #             bool_old = all_rendered_clusters_tensor[max_ind].sum(dim=0).bool()
        #             bool_new = all_new_clusters[i].sum(dim=0).bool()
        #             participating_old = responsible_gaussians_per_pixel * (bool_old.cuda() & ~bool_new.cuda())
        #             participating_old = torch.unique(participating_old)
        #             participating_old = participating_old[participating_old >= 0]
        #             self.assignments[participating_old.long(),old_cluster_ind[max_ind]] = False
        #             self.assignments[:,old_cluster_ind[max_ind]] = self.assignments[:,old_cluster_ind[max_ind]] | current_assignments[:,i]
                    
        #             del rendered_cluster_features[new_cid]
        #             #self.num_clusters -= 1
                    
        #             # add mask
        #             assert old_cluster_ind[max_ind.item()].item() in self.cluster_masks.keys()
        #             self.cluster_masks[old_cluster_ind[max_ind].item()].append((iter_time_idx, i))
                
        #         else:

                        
        #             self.assignments = torch.cat([self.assignments, current_assignments[:, i].unsqueeze(1)], dim=1)
        #             real_new_cluster_inds.append(self.num_clusters)
        #             rendered_cluster_features[self.num_clusters] = rendered_cluster_features.pop(new_cid)
        #             # rendered_cluster_features.remove
        #             # save masks
        #             #assert self.num_clusters not in self.cluster_masks.keys()
        #             self.cluster_masks[self.num_clusters] = [(iter_time_idx, i)]
        #             self.num_clusters += 1
                
        self.num_clusters = self.assignments.size(1)

        merged_cluster_ids = old_cluster_ind.tolist() + real_new_cluster_inds#new_cluster_ids 

        # re-cluster
        # if iter_time_idx != 0 and iter_time_idx % 10 == 0:
        #     self.recluster(params, cluster_input, iter_time_idx, sam_masks[0].size())
                    
        if iter_time_idx % 10 == 0:
            plot_cluster_language_association(rendered_cluster_features, None, curr_data['im'], merged_cluster_ids, out_file_prefix=f"visualization/cluster_visualization_{iter_time_idx}")

        self.assignments = self.assignments.to('cpu')
        
        torch.cuda.empty_cache()

    def recluster(self, params, cluster_input, iter_time_idx, image_size, not_pruned):
        if not_pruned is None:
            return
        self.assignments = self.assignments[not_pruned.squeeze()]
        # first get all rendered images
        H, W = image_size
        gaussians_contributions_per_frame = torch.zeros((iter_time_idx + 1, H, W))
        for i in range(iter_time_idx):
            
            with torch.no_grad():
                transformed_gaussians = get_transformed_gaussians(params, False, True, False, i)
                        
                rendervar_sil = transformed_params2depthplussilhouette(params, self.cam_data_collection[i]['w2c'],
                                                                    transformed_gaussians)

                _, _, _, ga_contr, _ = Renderer(raster_settings=self.cam_data_collection[i]['cam'])(**rendervar_sil)
                
                gaussians_contributions_per_frame[i] = ga_contr[0]
        
        # go through all clusters
        new_cluster_masks = {}
        new_assignments = torch.zeros_like(self.assignments)#-1 * torch.ones_like(self.assignments)
        #sam_masks = cluster_input['sam_masks'][iter_time_idx]
        for new_cluster_id, (cidx, mask_list) in enumerate(self.cluster_masks.items()):
            new_cluster_masks[new_cluster_id] = mask_list
            for (mask_time_idx, mask_ind) in mask_list:
                
                participating_gaussians = gaussians_contributions_per_frame[mask_time_idx] * cluster_input['sam_masks'][mask_time_idx][mask_ind]
                participating_gaussians = torch.unique(participating_gaussians)
                participating_gaussians = participating_gaussians[participating_gaussians != -1]

                new_assignments[participating_gaussians.long(), new_cluster_id] = True
        
        self.cluster_masks = new_cluster_masks
        self.num_clusters = new_assignments.size(1)
        #self.cluster_id = self.num_clusters     
        self.assignments = new_assignments
        
    def merge_at_view(self, params, iter_data, current_time_idx, iter_time_idx, cluster_input, image_size):
        
        # curr_data = self.cam_data_collection[iter_time_idx]
        # first render all clusters separately at current view
        H, W = image_size
        temp_rendered_cluster = torch.zeros((self.assignments.size(1), H, W))
        
        for i in range(self.assignments.size(1)):
        
            column = self.assignments[:, i]
            collected_gaussians = torch.where(column == True) 
        
            selected_params = {'means3D': params['means3D'][collected_gaussians],
                            #'means2D': params['means2D'][filtered_gaussians_idx],
                            'rgb_colors': params['rgb_colors'][collected_gaussians],
                                'unnorm_rotations': params['unnorm_rotations'][collected_gaussians],
                                'logit_opacities': params['logit_opacities'][collected_gaussians],
                                'log_scales': params['log_scales'][collected_gaussians],
                                # 'semantic_features': params['semantic_features'][collected_gaussians],
                                'cam_unnorm_rots': params['cam_unnorm_rots'],
                                'cam_trans': params['cam_trans']}
            
            # render cluster
            with torch.no_grad():
                transformed_gaussians = get_transformed_gaussians(params, False, True, False, iter_time_idx)
                selected_transformed_gaussians = {}
                for key, item in transformed_gaussians.items():
                    selected_transformed_gaussians[key] = item[collected_gaussians]
                        
                rendervar_sil = transformed_params2depthplussilhouette(selected_params, curr_data['w2c'],
                                                                    selected_transformed_gaussians)

                rend_sil, _, _, _, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar_sil)
        
            temp_rendered_cluster[i] = rend_sil.sum(dim=0) != 0
            
        ious = calculate_iou(temp_rendered_cluster, temp_rendered_cluster)
        ious = ious.fill_diagonal_(0)
        
        size_cluster_masks = (temp_rendered_cluster != 0).sum(dim=[1,2])
        sizes_i = size_cluster_masks.unsqueeze(0)
        sizes_j = size_cluster_masks.unsqueeze(1)
        size_ratios = torch.min(sizes_i, sizes_j) / torch.max(sizes_i, sizes_j)
        
        #top_k_ious = torch.topk(ious, 3, dim=1)
        
        adjacency_matrix = torch.logical_and(ious > 0.9, size_ratios > 0.95)#(ious > 0.9) and (size_ratios > 0.95)
        
        visited_masks = torch.zeros(self.assignments.size(1), dtype=torch.bool)
        
        components = []
        
        def dfs(node, component):
            # DFS to find all connected nodes
            stack = [node]
            while stack:
                n = stack.pop()
                if not visited_masks[n]:
                    visited_masks[n] = True
                    component.append(n)
                    neighbors = torch.nonzero(adjacency_matrix[n]).flatten().tolist()
                    stack.extend(neighbors)

        for i in range(self.assignments.size(1)):
            if not visited_masks[i]:
                component = []
                dfs(i, component)
                components.append(component)
        
        merged_masks = []
        number_plotted = 0
        clusters_to_keep_ind = torch.ones((self.assignments.size(1)), dtype=torch.bool)
        for component in components:
            # components need to be at least two elements/clusters to be merged
            if len(component) < 2:
                continue
            
            for i in component:
                plot_single_cluster(temp_rendered_cluster[i][None,:], curr_data['im'], f'temp{i}')
            
            # if they have been grouped, merge the cluster
            clusters_to_keep_ind[component[1:]] = False
            assignments_to_add = self.assignments[:,component[1:]]
            #self.assignments[assignments_to_add != -1] = self.assignments[]
            assignments_to_add = torch.any(assignments_to_add, dim=1)
            self.assignments[:,component[0]] |= assignments_to_add#assignments_to_add.sum(dim=1)
            
            for removed_cluster in component[1:]:
                self.cluster_masks[component[0]] += self.cluster_masks[removed_cluster]
                # self.cluster_masks.pop(removed_cluster)
            
        self.assignments = self.assignments[:,clusters_to_keep_ind]
        self.num_clusters = self.assignments.size(1) 
        
        cum_sum_keep_cluster = (~clusters_to_keep_ind).cumsum(dim=0)
        self.cluster_masks = {i - cum_sum_keep_cluster[i].item(): self.cluster_masks[i] for i in self.cluster_masks.keys() if clusters_to_keep_ind[i]}
        
        print("iou computed")
        
        
def get_clips_for_cluster(clips, cluster):
    unique_cluster_id_values = torch.unique(cluster)
    cluster_ids = unique_cluster_id_values[unique_cluster_id_values != -1] 
    cluster_ids = [val.item() for val in cluster_ids]
    return [clips[cid] for cid in cluster_ids], cluster_ids


def debug_assignments(self_assignments):
    num_gaussians, num_clusters = self_assignments.shape  # Get dimensions
    
    print(f"\n🔍 DEBUG INFO for self.assignments 🔍")
    print(f"➡ Number of Gaussians: {num_gaussians}")
    print(f"➡ Number of Clusters: {num_clusters}")

    # Dictionary to store unique values per column
    # unique_values_per_column = {}
    # count_per_column = {}

    # for col in range(num_clusters):
    #     unique_values = torch.unique(self_assignments[:, col])
    #     unique_values_per_column[col] = unique_values.tolist()
    #     count_per_column[col] = len(unique_values)

    # print(f"\n📌 Unique values per cluster (column-wise):")
    # for col, values in unique_values_per_column.items():
    #     print(f"  - Cluster {col}: {values}")
    

    # print(f"\n📊 Number of unique values per cluster:")
    # for col, count in count_per_column.items():
    #     print(f"  - Cluster {col}: {count} unique values")