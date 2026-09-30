"""
Standalone benchmark ground truth for TEILP core reasoning components.

This file consolidates model-side logic for:
- differentiable temporal logical rule reasoning,
- conditional time probability construction,
- TEKG reference-event graph construction.

Training loops, CLI code, multiprocessing random-walk scripts, dataset loaders,
checkpoint I/O, and evaluation orchestration are intentionally excluded.
"""

import itertools
from collections import Counter

import numpy as np
import tensorflow as tf


# --- [Original file: src/utlis.py] ---
def str_tuple(e):
    return str(tuple(e))


def gaussian_pdf(x, mu, std):
    """
    Calculate the probability density function (PDF) of a Gaussian distribution
    with mean mu and standard deviation std at the point x.
    """
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * std**2))


def calculate_TR(interval1, interval2):
    interval1 = [min(interval1), max(interval1)]
    interval2 = [min(interval2), max(interval2)]
    if 9999 in interval1 or 9999 in interval2:
        return 'ukn'
    if interval1[0] < interval2[0] and interval1[1] <= interval2[0]:
        return 'bf'
    if interval1[0] >= interval2[1] and interval1[1] > interval2[1]:
        return 'af'
    return 'touch'


def calculate_TR_mat_ver(interval1, interval2):
    # Assuming interval1 and interval2 are numpy arrays of shape (batch_size, 1) or (batch_size, 2)
    interval1 = np.array(interval1)
    interval2 = np.array(interval2)

    def process_intervals(interval):
        if interval.shape[1] == 1:
            return np.column_stack((interval[:, 0], interval[:, 0]))
        else:
            return np.column_stack((np.min(interval, axis=1), np.max(interval, axis=1)))
    
    # Process both intervals
    interval1_processed = process_intervals(interval1)
    interval2_processed = process_intervals(interval2)
    
    # Check for presence of 9999 in any row of either matrix
    invalid_rows = np.any((interval1_processed == 9999) | (interval2_processed == 9999), axis=1)
    
    # Set default TR to 2 (touching)
    result = np.ones(interval1_processed.shape[0], dtype=int) * 2
    
    # Check conditions for each row
    result[np.logical_and(interval1_processed[:, 0] < interval2_processed[:, 0], interval1_processed[:, 1] <= interval2_processed[:, 0])] = 1
    result[np.logical_and(interval1_processed[:, 0] >= interval2_processed[:, 1], interval1_processed[:, 1] > interval2_processed[:, 1])] = 3
    
    # Set invalid rows to 0
    result[invalid_rows] = 0
    
    return result


def cal_timegap(timestamp1, timestamp2, notation_invalid=9999):
    if timestamp1 == 9999 or timestamp2 == 9999:
        return notation_invalid
    else:
        return timestamp1 - timestamp2


# --- [Original file: src/model.py] ---
class Learner(object):
    def __init__(self, option, data):
        self.seed = option.seed
        self.num_step = option.num_step
        self.num_layer = option.num_layer
        self.rnn_state_size = option.rnn_state_size
        
        self.norm = not option.no_norm
        self.thr = option.thr
        self.dropout = option.dropout

        self.num_relation = data['num_rel']
        self.num_TR = data['num_TR']
        self.num_rule = option.num_rule
        self.num_timestamp = len(data['timestamp_range'])
        self.num_query = data['num_query']
        self.query_embed_size = option.query_embed_size
        

        self.flag_int = option.flag_interval
        self.flag_rel_TR_split = option.different_states_for_rel_and_TR
        self.flag_ruleLen_split = option.flag_ruleLen_split_ver
        self.flag_state_vec_enhance = option.flag_state_vec_enhancement
        self.flag_acceleration = option.flag_acceleration

        # weight for the state vector enhancement (w/ shallow layers)
        # final state = gamma * state + (1-gamma) * enhance_state
        self.gamma = 0.7

        # To make the learning more stable, we scale the final prob dynamically.
        # If the prob is greater than self.prob_scaling_factor[0], we scale it by self.prob_scaling_factor[1].
        self.prob_scaling_factor = [0.1, 0.5]
 
        np.random.seed(self.seed)

        if option.flag_ruleLen_split_ver:
            print('Todo: ruleLen_split_ver')
            pass

    def _random_uniform_unit(self, r, c):
        bound = 6./ np.sqrt(c)
        init_matrix = np.random.uniform(-bound, bound, (r, c))
        init_matrix = np.array(map(lambda row: row / np.linalg.norm(row), init_matrix))
        return init_matrix


    def _scale_final_prob(self, prob):
        # Calculate if any entry is greater than self.prob_scaling_factor[0]
        condition = tf.reduce_any(tf.greater(prob, self.prob_scaling_factor[0]))

        # Use tf.cond to set prob_scaling_factor
        prob_scaling_factor = tf.cond(
            condition,
            lambda: tf.constant(self.prob_scaling_factor[1]),
            lambda: tf.constant(self.prob_scaling_factor[0])
        )

        # scaling the loss to make the learning more stable.
        prob /= prob_scaling_factor
        return prob


    def _time_prediction(self, state_vec, query_time_dist, target_entities):
        '''
        Given the final state vector, we predict the query time.
        '''
        # The state vector is about the probability we arrive at different reference nodes. 
        # Given each reference node, we can calculate a probability for the query time.
        # shape change: (batch_size, num_nodes, 1) * (batch_size, num_nodes, num_timestamp) -> (batch_size, num_nodes, num_timestamp)
        # For training, since we know the target timestamp, we can simply the calculatation as:
        # shape change: (batch_size, num_nodes) * (batch_size, num_nodes) -> (batch_size, num_nodes)
        pred = state_vec * query_time_dist
        
        # We require the path to be cyclic. And the last TR can only be ukn since we don't know the query time.
        pred = tf.transpose(pred)
        pred = tf.sparse_tensor_dense_matmul(self.connectivity_TR[0], pred)
        pred = tf.transpose(pred)
        pred = tf.reduce_sum(target_entities * pred, axis=1, keep_dims=True)

        # To make the learning more stable, we scale the final prob.
        # All the operations are the same as above except without the multiplication of query_time_dist.
        norm = tf.identity(state_vec)
        norm = tf.transpose(norm)
        norm = tf.sparse_tensor_dense_matmul(self.connectivity_TR[0], norm)
        norm = tf.transpose(norm)
        norm = 1e-20 + tf.reduce_sum(target_entities * norm, axis=1, keep_dims=True)

        return pred/norm


    def _calculate_loss(self, pred):
        return - tf.reduce_sum(tf.log(tf.maximum(pred, self.thr)), 1)


    def _selection_block(self, state_vec, attn, trans_mat, choice_ls):
        '''
        Select elements (either relations or TRs) during transition via a weighted sum.
        '''
        state_vec = tf.transpose(state_vec)

        trans_results = []
        for idx in choice_ls:  # now connectivity_rel is a diagonal matrix which shows the predicate of nodes
            product = tf.sparse_tensor_dense_matmul(trans_mat[idx], state_vec)
            product = tf.transpose(product)
            trans_results.append(attn[idx] * product)
        
        added_trans_results = tf.add_n(trans_results)
        
        if self.norm:
            added_trans_results /= tf.maximum(self.thr, tf.reduce_sum(added_trans_results, axis=1, keep_dims=True))
                
        return added_trans_results


    def _obtain_state_vec_enhance(self, num_sample, num_entity):
        '''
        Use shallow layers to enhance the state vector.
        notation: 0: last_event, 1: first_event (different from fast ver)
        '''
        attn_rule = tf.nn.embedding_lookup(self.attn_rule_embed, self.query_rels_flatten)
        self.attn_rule = tf.nn.softmax(attn_rule, axis=2) # shape: (flatten_batch_size, 2, num_rule)

        # We calculate the probability we arrive at different reference nodes from shallow rule scores.
        #    self.random_walk_ind: the index of rule for each ref event
        #           shape:(num_ref_events, 2, num_rules) [2 channels: we distinguish the nodes as last or first events]
        #    res shape: (flatten_batch_size, )
        refNode_probs = [tf.reduce_sum(self.random_walk_ind[:, i, :] * self.attn_rule[:, i, :], axis=1, keep_dims=False) for i in range(2)]
        
        # We obtain the state vector enhancement by assigning the probability to certain index of the state vector.
        # res shape: (batch_size, num_entity)
        state_vec_enhance = [tf.scatter_nd(self.refNode_index, refNode_probs[i], [num_sample, num_entity]) for i in range(2)]
        
        # normalize the state vector enhancement
        state_vec_enhance = [x/(1e-20 + tf.reduce_sum(x, axis=1, keep_dims=True)) for x in state_vec_enhance]
        
        return state_vec_enhance


# --- [Original file: src/gadgets.py] ---
class Data_Processor(object):
    def _calculate_time_prob_dist(self, edge, timestamp_range, targ_interval, stat_ls, idx_ls, mode, flag_time_shift=False, probs_normalization=True, flag_interval=True):
        '''
        Calculate the probability distribution of the time gap.

        Parameters:
            edge: tuple, edge
            timestamp_range: np.array, timestamp range
            targ_interval: np.array, target interval
            stat_ls: list, statistics
            idx_ls: list, indices
            mode: str, mode
            flag_time_shift: bool, whether to use time shift
            probs_normalization: bool, whether to normalize the probabilities
            flag_interval: bool, whether to use interval

        Returns:
            flag_success: bool, whether the calculation is successful
            cur_probs: np.array, current probabilities
        '''
        idx_ts_or_te, idx_first_or_last_event, idx_ref_time_ts_or_te = idx_ls

        # Find the index of the statistics (which is a composite vector)
        idx_stat = 4*idx_ts_or_te + 2*idx_first_or_last_event + idx_ref_time_ts_or_te if flag_interval else idx_first_or_last_event

        # edge: (s, r, ts, te, o) or (s, r, ts, o)
        refTime = edge[2:4][idx_ref_time_ts_or_te] if flag_interval else edge[2:3][idx_ref_time_ts_or_te]   
        delta_t = timestamp_range - refTime
        targ_time = targ_interval[idx_ts_or_te]

        Gau_mean = stat_ls[idx_stat*2]
        Gau_std = stat_ls[idx_stat*2+1]
        
        if 9999 in [refTime, Gau_mean, Gau_std]:
            # invalid time exists
            return False, []

        cur_probs = gaussian_pdf(delta_t, Gau_mean, Gau_std)
        
        if flag_time_shift:
            cur_probs[delta_t < 0] = 0

        if len(cur_probs[np.isnan(cur_probs)])>0 or len(cur_probs[np.isinf(cur_probs)])>0 or sum(cur_probs) == 0:
            return False, []

        if probs_normalization:
            cur_probs /= np.sum(cur_probs)
        
        cur_probs = np.around(cur_probs, decimals=6)

        if mode == 'Train':
            # use ground truth
            cur_probs = cur_probs[delta_t.tolist().index(targ_time - refTime)]

        return True, cur_probs


    def _calculate_event_distribution(self, data, flag_interval):
        '''
        Given a rule, count the distribution of the first and last events. 
        Found that other positions are not so useful.

        Parameters:
            data: list, data

        Returns:
            first_position_distribution: dict, distribution of the first event
            last_position_distribution: dict, distribution of the last event
        '''
        first_position_counter = Counter()
        last_position_counter = Counter()

        for event_ls in data:
            first_event, last_event = tuple(event_ls[0]), tuple(event_ls[1])
            
            # remove unknown time events
            # 9999 is the placeholder for unknown time
            # Event format: (s, r, ts, te, o)
            if flag_interval:
                if first_event[2] != 9999 and first_event[3] != 9999:  
                    first_position_counter[first_event] += 1
    
                if last_event[2] != 9999 and last_event[3] != 9999:    
                    last_position_counter[last_event] += 1
            else:
                if first_event[2] != 9999:  
                    first_position_counter[first_event] += 1
    
                if last_event[2] != 9999:    
                    last_position_counter[last_event] += 1

        total_first = sum(first_position_counter.values())
        total_last = sum(last_position_counter.values())
        
        first_position_distribution = {k: v*1. / total_first for k, v in first_position_counter.items()}
        last_position_distribution = {k: v*1. / total_last for k, v in last_position_counter.items()}

        return [first_position_distribution, last_position_distribution]


    def _create_probs_dict(self, walk_res, query_time, pattern_ls, ts_stat_ls, te_stat_ls, timestamp_range, 
                           flag_rule_split, mode, flag_time_shift, probs_normalization, flag_interval):
        '''
        Given a sample, create a dictionary for storing the probabilities of the time gap.

        Parameters:
            walk_res: walk_res for the current sample
            (global) pattern_ls: list of all possible rules given the query relation
            (global) ts_stat_ls: list of statistics for the query start time given the query relation
            (global) te_stat_ls: list of statistics for the query end time given the query relation
            timestamp_range: np.array, timestamp range
            flag_rule_split: bool, whether to split the rules
            mode: str, mode
            flag_time_shift: bool, whether to use time shift
            probs_normalization: bool, whether to normalize the probabilities
            flag_interval: bool, whether to use interval

        Returns:
            probs: dict, probabilities of the time gap
        '''
        valid_rules = [p for p in walk_res if p in pattern_ls]
        if len(valid_rules) == 0:
            return None
        
        probs = {'0':{}, '1':{}}  # [first event, last event]
        for p in valid_rules:
            # for each rule, calculate the probability of the time gap
            p_idx = pattern_ls.index(p)
            ruleLen = len(p.split(' '))//2
            cur_stat_ls = ts_stat_ls[p_idx]
            if flag_interval: 
                cur_stat_ls += te_stat_ls[p_idx]

            events_for_cur_rule = []
            for walk in walk_res[p]['edge_ls']:
                events_for_cur_rule.append([walk[idx] for idx in [0, -1]])  # we only consider the first and last event
            
            events_for_cur_rule = self._calculate_event_distribution(events_for_cur_rule, flag_interval)

            for idx_event_pos in [0, 1]:
                for edge in events_for_cur_rule[idx_event_pos]:
                    alpha = events_for_cur_rule[idx_event_pos][edge]            
                    if str_tuple(edge) not in probs[str(idx_event_pos)]:
                        probs[str(idx_event_pos)][str_tuple(edge)] = {str(i): [] for i in range(4)} if not flag_rule_split else \
                                                                     {str(i): {str(rLen): [] for rLen in range(1, 6)} for i in range(4)}

                    for idx_query_time in [0, 1]:
                        for idx_ref_time in [0, 1]:
                            if (not flag_interval) and ((idx_query_time !=0) or (idx_ref_time != 0)):
                                continue
                            flag_success, cur_probs = self._calculate_time_prob_dist(edge, timestamp_range, query_time, cur_stat_ls, 
                                                                                        [idx_query_time, idx_event_pos, idx_ref_time], 
                                                                                        mode, flag_time_shift, probs_normalization, flag_interval)
                            if not flag_success:
                                cur_probs = np.array(1./len(timestamp_range)) if mode == 'Train' else \
                                            np.array([1./len(timestamp_range) for _ in range(len(timestamp_range))])
                            
                            cur_probs = (cur_probs * alpha).tolist()
                            cur_probs = [cur_probs, p_idx]  # Add rule idx for tracking.

                            idx_composite = 2*idx_query_time + idx_ref_time
                            if flag_rule_split:
                                probs[str(idx_event_pos)][str_tuple(edge)][str(idx_composite)][str(ruleLen)].append(cur_probs)                                    
                            else:
                                probs[str(idx_event_pos)][str_tuple(edge)][str(idx_composite)].append(cur_probs)
        return probs


class Base(object):
    def __init__(self, option, data):
        self.option = option
        self.data = data

    def _merge_list_inside(self, ori_ls):
        '''
        Given a list, where each element has the same number of lists, we want to merge the inside lists at the same position.
        E.g. [[ls1, ls2], [ls3, ls4]] -> [ls1 + ls3, ls2 + ls4]
        '''
        if len(ori_ls) == 0:
            return []
        
        merged_ls = []
        for j in range(len(ori_ls[0])):
            output = []
            for i in range(len(ori_ls)):
                output += ori_ls[i][j]
            merged_ls.append(np.array(output))
        return merged_ls


    def _merge_array_inside(self, ori_ls):
        '''
        Given a list, where each element has the same number of arrays, we want to merge the inside arrays at the same position.
        E.g. [[array1, array2], [array3, array4]] -> [np.vstack([array1, array3]), np.vstack([array2])]
        '''
        if len(ori_ls) == 0:
            return []
        
        merged_ls = []
        for j in range(len(ori_ls[0])):
            output = []
            for i in range(len(ori_ls)):
                output.append(ori_ls[i][j])
            merged_ls.append(np.vstack(output))
        return merged_ls


# --- [Original file: src/Graph.py] ---
class TEKG_fast_ver(Base):
    def __init__(self, option, data, call_by_TEKG=False):
        super(TEKG_fast_ver, self).__init__(option, data)
        # call_by_TEKG: 
        #   In TEKG, we need to distinguish one event at different event pos, i.e., 
        #   if an event is both the first and the last event, we consider it as two events.
        #   Also, we need to prepare all the data during training if call_by_TEKG is True.
        self.call_by_TEKG = call_by_TEKG

    def _build_refNode_structures(self, walk_res, mode, selected_rule_ls=None, rule_scores=None, refType_scores=None):
        '''
        For each sample, populates refNode_probs and refNode_rule_idx based on random_walk_res
        '''
        flag_valid = False
        refNode_probs, refNode_rule_idx, preds = {}, {}, {}
        for idx_event_pos in [0,1]:
            for edge in walk_res[str(idx_event_pos)]:
                if (eval(edge)[2] == 9999) or (self.option.flag_interval and eval(edge)[3] == 9999):
                    # we don't need unknown time events as reference events
                    continue
                if edge not in refNode_probs:
                    refNode_probs[edge] = {0: {0: {0:[], 1:[]}, 1: {0:[], 1:[]}}, 1: {0: {0:[], 1:[]}, 1: {0:[], 1:[]}}}  # [idx_event_pos][idx_query_time][idx_ref_time]
                  
                    refNode_rule_idx[edge] = []
                    if mode == 'Test' and (not self.call_by_TEKG):
                        if self.option.flag_interval:
                            preds[edge] = {0: 0, 1: 0}  # query ts or te
                        else:
                            preds[edge] = {0: 0} # query ts

                # for each event, we have a list of probabilities
                for idx_query_time in [0,1]:
                    for idx_ref_time in [0,1]:
                        if (not self.option.flag_interval) and ((idx_query_time != 0) or (idx_ref_time != 0)):
                            continue

                        idx_complete = idx_query_time*2 + idx_ref_time
                        probs = walk_res[str(idx_event_pos)][edge][str(idx_complete)]

                        # Given the query, the event pos and the reference event, more than one rules are satisfied.
                        for prob_dict in probs:
                            if (selected_rule_ls is not None) and (prob_dict[1] not in selected_rule_ls):
                                continue
                            
                            refNode_probs[edge][idx_event_pos][idx_query_time][idx_ref_time].append(prob_dict)
                            refNode_rule_idx[edge].append(prob_dict[1])
                            
                            if mode == 'Test' and (rule_scores is not None) and (refType_scores is not None) and (not self.call_by_TEKG):
                                if self.option.flag_interval:
                                    # refType_scores: [(first_event_ts, first_event_te), (last_event_ts, last_event_te), (first_event, last_event)]
                                    prob_event_pos = refType_scores[3*idx_query_time + 2][idx_event_pos]
                                    prob_ref_time = refType_scores[3*idx_query_time + idx_event_pos][idx_ref_time]
                                    preds[edge][idx_query_time] += prob_event_pos * prob_ref_time * rule_scores[prob_dict[1]] * np.array(prob_dict[0])
                                else:
                                    prob_event_pos = refType_scores[0][idx_event_pos]
                                    preds[edge][idx_query_time] += prob_event_pos * rule_scores[prob_dict[1]] * np.array(prob_dict[0])

                            flag_valid = True

        return flag_valid, refNode_probs, refNode_rule_idx, preds


    def _update_outputs(self, refNode_probs, refNode_rule_idx, query_relations, idx, valid_sample_idxs, 
                              refNode_nums, query_rel_flatten, probs, refEdges, preds, mode):
        '''
        Updates the output structures with processed data from refNode_probs and refNode_rule_idx
        '''
        # This is a placeholder for the actual logic
        if self.option.flag_ruleLen_split_ver:
            print('Todo')
            pass
        
        if mode == 'Test' and (len(preds) > 0) and (not self.call_by_TEKG):
            # Given the sample, merge probs from different events.
            final_preds = {0: 0, 1: 0} if self.option.flag_interval else {0: 0}
            for edge in preds:
                for idx_query_time in [0,1]:
                    if not self.option.flag_interval and idx_query_time != 0:
                        continue
                    final_preds[idx_query_time] += preds[edge][idx_query_time]
              
            valid_sample_idxs.append(idx)     
            return final_preds
 

        num_valid_edge = 0
        for edge in refNode_rule_idx:
            # refNode_rule_idx[edge]: num of different rules satisfied for the current edge (idx_event_pos can be 0 or 1)
            if len(refNode_rule_idx[edge]) == 0:
                continue
            
            cnt_edge_num = 0
            probs_with_rule_idx = []
            for idx_query_time in [0, 1]:
                for idx_event_pos in [0, 1]:
                    num_rules = 0
                    for idx_ref_time in [0, 1]:
                        if (not self.option.flag_interval) and ((idx_query_time != 0) or (idx_ref_time != 0)):
                            continue
                        cur_probs = refNode_probs[edge][idx_event_pos][idx_query_time][idx_ref_time]
                        num_rules += len(cur_probs)
                        probs_with_rule_idx.append([[len(probs), prob_dict[1], prob_dict[0]] for prob_dict in cur_probs])  # [idx_event_in_batch, idx_rule, prob]

                    # If there are rules for the current event_pos, we add the edge to refEdges.
                    # We only consider it for tqs since we don't want to add the same edge multiple times.
                    if idx_query_time == 0 and num_rules > 0:
                        refEdges.append([edge, idx_event_pos])
                        cnt_edge_num += 1
            
            # If we do not want to distinguish the event pos, we only consider the edge once.
            cnt_edge_num = min(cnt_edge_num, 1) if not self.call_by_TEKG else cnt_edge_num
            num_valid_edge += cnt_edge_num

            probs.append(probs_with_rule_idx)  # [idx_event_in_batch, idx_rule, prob] * 8 if flag_interval else * 2


        if num_valid_edge > 0:
            # update these global variables.
            valid_sample_idxs.append(idx)
            query_rel_flatten += [query_relations[idx]] * num_valid_edge
            refNode_nums.append(num_valid_edge)
        
        return None


    def _obtain_one_hot_form(self, source_ls):
        '''
        Convert the source_ls into one-hot form.
        source_ls: [num of rules in each sample], e.g. [2, 13]
        one_hot: one hot form of source_ls, e.g. [[1, 1, 0, 0, 0, ...], [0, 0, 1, 1, 1, ...]]
        '''
        refNode_num = 0
        num_nodes = sum(source_ls)
        one_hot = []
        for i in range(len(source_ls)): # len(source_ls): batch_size
            sources = np.zeros((num_nodes, ))
            sources[refNode_num: refNode_num + source_ls[i]] = 1
            one_hot.append(sources)
            refNode_num += source_ls[i]
            
        return one_hot


class TEKG(Base):
    def _build_connectivity_rel(self, nodes, nodes_idx, num_entity, num_rel):
        """
        Build the connectivity_rel mat based on current nodes.
        Nodes are augmented with inverse nodes.
        """
        connectivity_rel = {}
        for idx_rel in range(num_rel):
            # Find all the nodes that satisfy the relation.
            idx_nodes_cur_rel = nodes_idx[nodes[:, 1] == idx_rel].reshape((-1, 1))
            if idx_nodes_cur_rel.size == 0:
                connectivity_rel[idx_rel] = [[[0,0]], [0.], [num_entity, num_entity]]
            else:
                x, y = idx_nodes_cur_rel, idx_nodes_cur_rel
                connectivity_rel[idx_rel] = [np.hstack([x, y]).tolist(), [1.0 for _ in range(len(idx_nodes_cur_rel))], [num_entity, num_entity]]
        return connectivity_rel


    def _build_aug_nodes(self, nodes, num_entity):
        """
        Build the augmented nodes based on the current nodes.
        """
        nodes_inv = self._obtain_inverse_edges(nodes, self.data['num_rel'])
        nodes_aug = np.vstack((nodes, nodes_inv))
        nodes_idx = np.arange(len(nodes))
        nodes_idx_aug = np.hstack((nodes_idx, nodes_idx + num_entity//2))
        return nodes_aug, nodes_idx_aug


    def _build_connectivity_TR(self, nodes, nodes_idx, num_entity, num_TR):
        """
        Build the connectivity_TR mat based on current nodes.
        Nodes are augmented with inverse nodes.
        Different temporal relations (TRs) [0: ukn, 1: bf, 2: touch, 3: af]
        """
        connectivity_TR = {key: [] for key in range(num_TR)}    
        for entity in np.unique(np.hstack([nodes[:, 0], nodes[:, 2]])):
            # Find the node pairs that have the same entities as the subject and object.
            # Direction: b -> a
            b = nodes_idx[nodes[:, 2] == entity]
            a = nodes_idx[nodes[:, 0] == entity]
            combinations = np.array(list(itertools.product(a, b)))
            
            # Calculate the TRs for the node pairs.
            TRs = calculate_TR_mat_ver(nodes[combinations[:,0], 3:], nodes[combinations[:,1], 3:])

            # For different TRs, we store the corresponding node pairs with mask.
            for i in range(num_TR):
                mask = TRs == i if i else slice(None)
                connectivity_TR[i] += combinations[mask].tolist() 

        # Convert the output format.
        for idx_TR in range(num_TR):
            if connectivity_TR[idx_TR]:
                A = np.array(connectivity_TR[idx_TR])
                if A.size == 0:
                    connectivity_TR[idx_TR] = [[[0,0]], [0.], [num_entity, num_entity]]
                else:
                    A = np.unique(A, axis=0)
                    connectivity_TR[idx_TR] = [A.tolist(), [1.0 for _ in range(len(A))], [num_entity, num_entity]]
            else:
                connectivity_TR[idx_TR] = [[[0,0]], [0.], [num_entity, num_entity]]

        return connectivity_TR


    def _obtain_inverse_edges(self, edges, num_rel):
        """
        Inverse the edges and adjust relation IDs for symmetry.
        Edges: [entity1, relation, entity2, start_time, end_time]
        Inv edges: [entity2, inv_relation, entity1, start_time, end_time]
        """
        inv_edges = edges[:, [2, 1, 0, 3, 4]] if self.option.flag_interval else edges[:, [2, 1, 0, 3]]
        mask = edges[:, 1] < num_rel // 2
        inv_edges[mask, 1] += num_rel // 2
        inv_edges[~mask, 1] -= num_rel // 2
        return inv_edges


    def _select_probability(self, probs, mode, min_prob=1e-4):
        '''
        Select the probability for the current reference events (There might be multiple rules satisfied).
        '''
        if mode == 'Train':
            # min_prob: set minimum probability that make the training more stable.
            if self.option.prob_selection_for_training == 'max':
                return max(min_prob, max(probs))
            else:
                return max(min_prob, np.mean(probs))
        else:
            probs = np.array(probs) # shape: (num_rules, num_timestamp)
            return np.mean(probs, axis=0)
      

    def _calculate_distribution_score(self, extra_data, all_idx):
        '''
        Given all the related indices, we calculate the score for current distribution.
        '''
        data_idx, edge_idx, idx_query_time, idx_event_pos, idx_ref_time = all_idx
        final_state_vec, attn_refType = extra_data

        if final_state_vec is None or attn_refType is None:
            return 0.
        
        # We first select corresponding final state vector.
        selected_final_prob = final_state_vec[str((data_idx, idx_event_pos, edge_idx))]
        selected_attn_refType = attn_refType[str(data_idx)]
        
        if self.option.flag_interval:   
            # attn_refType: [tqs, tqe] X [(last_ts, last_te), (first_ts, first_te), (last_event, first_event)]
            score = selected_final_prob * selected_attn_refType[3*idx_query_time + 2][1-idx_event_pos] \
                                        * selected_attn_refType[3*idx_query_time + 1-idx_event_pos][idx_ref_time]
        else:
            # attn_refType: [(last_event, first_event)]
            score = selected_final_prob * selected_attn_refType[0][1-idx_event_pos]

        return score


    def _obtain_edge_idx_in_TEKG(self, edge, TEKG_nodes, TEKG_nodes_idx, num_entity, idx_event_pos):
        '''
        Given an edge, obtain its idex in TEKG.
        '''
        edge_idx = TEKG_nodes_idx[TEKG_nodes.tolist().index(edge)]

        # For the first event, we will do a reverse walk. Thus, we need to find the inv node in TEKG.
        if idx_event_pos == 0:
            edge_idx += num_entity//2
        return edge_idx


    def _update_event_probabilities(self, probs, idx_sample, data_idx, walk_res, num_entity, TEKG_nodes_aug, TEKG_nodes_idx_aug, mode, 
                                    stage, final_preds, extra_data):
        '''
        Given the walk res, update the probs for the current sample.
        '''
        flag_valid = 0
        ref_event_idx = {}
        preds = [0. for _ in range(1 + int(self.option.flag_interval))]
        for idx_event_pos in [0, 1]:
            # Format: walk_res[idx_event_pos][str_tuple(edge)][2*idx_query_time + idx_ref_time]
            for edge in walk_res[str(idx_event_pos)]:
                # The format we used in walk res is different from the original one.
                edge_reformatted = [eval(edge)[j] for j in [0,1,4,2,3]] if self.option.flag_interval else [eval(edge)[j] for j in [0,1,3,2]]

                # Obtain the edge index in TEKG.
                edge_idx = self._obtain_edge_idx_in_TEKG(edge_reformatted, TEKG_nodes_aug, TEKG_nodes_idx_aug, num_entity, idx_event_pos)

                for idx_query_time in [0, 1]:
                    for idx_ref_time in [0, 1]:
                        if (not self.option.flag_interval) and ((idx_query_time != 0) or (idx_ref_time != 0)):
                            continue
                        cur_probs = [item[0] for item in walk_res[str(idx_event_pos)][edge][str(2*idx_query_time + idx_ref_time)]]
                        if len(cur_probs) == 0:
                            continue
 
                        selected_prob = self._select_probability(cur_probs, mode)

                        # Use a composite index here to simplify the coding.
                        # When building the model, we first calculate the last event and then the first event, first tqs and then tqe.
                        idx_complete = 4*idx_query_time + 2*(1-idx_event_pos) + idx_ref_time if self.option.flag_interval else 1-idx_event_pos
                        
                        if mode == 'Train':
                            probs[idx_sample][idx_complete][edge_idx] = selected_prob
                        else:
                            # No need to update the probabilities for inference. Instead we use an online algorithm.    
                            if stage == 'obtain state vec':
                                ref_event_idx[(data_idx, idx_event_pos, edge_idx)] = 0.
                            else:
                                # During inference, we calculate the time prediction in an online manner.
                                all_idx = [data_idx, edge_idx, idx_query_time, idx_event_pos, idx_ref_time]
                                preds[idx_query_time] += selected_prob * self._calculate_distribution_score(extra_data, all_idx)
                        flag_valid = 1
        
        if flag_valid and stage == 'time prediction':                
            final_preds.append(preds)

        return flag_valid, ref_event_idx


if __name__ == "__main__":
    import types

    np.random.seed(42)

    passed = 0
    failed = 0

    def check(test_name, condition, detail=""):
        global passed, failed
        if bool(condition):
            passed += 1
            print(f"  [{test_name}] PASS")
        else:
            failed += 1
            suffix = f" - {detail}" if detail else ""
            print(f"  [{test_name}] FAIL{suffix}")

    def skip_checks(count, reason):
        global failed
        failed += count
        print(f"  [skipped {count} check(s)] FAIL - {reason}")

    def make_option(flag_interval=True):
        return types.SimpleNamespace(
            flag_interval=flag_interval,
            flag_ruleLen_split_ver=False,
            prob_selection_for_training='max',
        )

    if not hasattr(tf, "sparse_tensor_dense_matmul") and hasattr(tf, "sparse"):
        tf.sparse_tensor_dense_matmul = tf.sparse.sparse_dense_matmul
    _tf_reduce_sum = tf.reduce_sum

    def _reduce_sum_compat(input_tensor, axis=None, keep_dims=None, keepdims=None, name=None):
        if keep_dims is not None and keepdims is None:
            keepdims = keep_dims
        return _tf_reduce_sum(input_tensor, axis=axis, keepdims=keepdims, name=name)

    tf.reduce_sum = _reduce_sum_compat

    print("=" * 70)
    print("TEILP reproduction benchmark: temporal logical reasoning components")
    print("Automated Test Suite - 8 ablated targets")
    print("=" * 70)
    print()

    print("-" * 60)
    print("[Test 1/5] temporal utility functions")
    try:
        check("gaussian pdf output not None", gaussian_pdf(np.array([0.0, 1.0]), 0.0, 1.0) is not None)
        pdf = gaussian_pdf(np.array([0.0, 1.0]), 0.0, 1.0)
        check("gaussian pdf shape", pdf.shape == (2,), f"got {pdf.shape}")
        check("gaussian pdf finite positive", np.isfinite(pdf).all() and np.all(pdf > 0))
        check("calculate_TR before", calculate_TR([1, 2], [3, 4]) == 'bf')
        check("calculate_TR after", calculate_TR([5, 6], [3, 4]) == 'af')
        check("calculate_TR unknown", calculate_TR([9999, 9999], [3, 4]) == 'ukn')
        mat_tr = calculate_TR_mat_ver(np.array([[1, 2], [5, 6], [9999, 9999]]), np.array([[3, 4], [3, 4], [1, 2]]))
        check("calculate_TR_mat_ver shape", mat_tr.shape == (3,), f"got {mat_tr.shape}")
        check("calculate_TR_mat_ver values", mat_tr.tolist() == [1, 3, 0], f"got {mat_tr.tolist()}")
    except Exception as exc:
        skip_checks(8, f"utility test raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 2/5] Data_Processor probability construction")
    try:
        processor = Data_Processor()
        timestamp_range = np.arange(1998, 2003)
        edge = (0, 1, 2000, 2001, 2)
        stat_ls = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
        ok, probs = processor._calculate_time_prob_dist(
            edge, timestamp_range, [2000, 2001], stat_ls, [0, 0, 0],
            mode='Test', flag_time_shift=False, probs_normalization=True, flag_interval=True
        )
        check("time prob output not None", probs is not None)
        check("time prob success", ok is True)
        if ok and probs is not None:
            check("time prob shape", probs.shape == (5,), f"got {getattr(probs, 'shape', None)}")
            check("time prob normalized", abs(float(np.sum(probs)) - 1.0) < 1e-5, f"sum {np.sum(probs)}")
            check("time prob finite", np.isfinite(probs).all())
        else:
            skip_checks(3, "time probability returned invalid result")

        ok_train, prob_train = processor._calculate_time_prob_dist(
            edge, timestamp_range, [2000, 2001], stat_ls, [0, 0, 0],
            mode='Train', flag_time_shift=False, probs_normalization=True, flag_interval=True
        )
        check("train time prob scalar", np.isscalar(prob_train), f"got {type(prob_train)}")
        check("train time prob positive", float(prob_train) > 0)

        walk_res = {
            "1 2": {
                "edge_ls": [
                    [[10, 5, 2000, 2000, 11], [11, 6, 2001, 2001, 12]],
                    [[10, 5, 2000, 2000, 11], [12, 6, 2002, 2002, 13]],
                ]
            }
        }
        probs_dict = processor._create_probs_dict(
            walk_res, [2001, 2002], ["1 2"], [stat_ls], [stat_ls],
            timestamp_range, flag_rule_split=False, mode='Test',
            flag_time_shift=False, probs_normalization=True, flag_interval=True
        )
        check("probs dict not None", probs_dict is not None)
        if probs_dict is not None:
            check("probs dict event positions", set(probs_dict.keys()) == {'0', '1'})
            check("probs dict has first event", len(probs_dict['0']) == 1)
            check("probs dict has last events", len(probs_dict['1']) == 2)
            first_edge = next(iter(probs_dict['0']))
            check("probs dict composite keys", set(probs_dict['0'][first_edge].keys()) == {'0', '1', '2', '3'})
            check("probs dict rule payload", probs_dict['0'][first_edge]['0'][0][1] == 0)
        else:
            skip_checks(5, "probability dictionary returned None")
    except Exception as exc:
        skip_checks(15, f"Data_Processor test raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 3/5] TEKG_fast_ver reference-node structures")
    try:
        graph = TEKG_fast_ver(make_option(flag_interval=True), data={}, call_by_TEKG=False)
        edge_a = str((1, 2, 2000, 2001, 3))
        edge_unknown = str((4, 2, 9999, 9999, 5))
        walk_res = {
            '0': {edge_a: {'0': [[[0.2, 0.8], 1]], '1': [], '2': [], '3': []},
                  edge_unknown: {'0': [[[0.5, 0.5], 2]], '1': [], '2': [], '3': []}},
            '1': {edge_a: {'0': [], '1': [[[0.1, 0.9], 3]], '2': [], '3': []}},
        }
        rule_scores = [0.1, 0.7, 0.2, 0.5]
        refType_scores = [[0.6, 0.4], [0.3, 0.7], [0.55, 0.45],
                          [0.4, 0.6], [0.2, 0.8], [0.5, 0.5]]
        flag_valid, refNode_probs, refNode_rule_idx, preds = graph._build_refNode_structures(
            walk_res, mode='Test', selected_rule_ls=[1, 3],
            rule_scores=rule_scores, refType_scores=refType_scores
        )
        check("refNode valid flag", flag_valid is True)
        check("unknown edge skipped", edge_unknown not in refNode_probs)
        check("refNode edge retained", edge_a in refNode_probs)
        check("refNode rule ids", sorted(refNode_rule_idx[edge_a]) == [1, 3], f"got {refNode_rule_idx[edge_a]}")
        check("refNode preds interval keys", set(preds[edge_a].keys()) == {0, 1})
        check("refNode pred vector shape", np.array(preds[edge_a][0]).shape == (2,))

        valid_sample_idxs, refNode_nums, query_rel_flatten, probs, refEdges = [], [], [], [], []
        result = graph._update_outputs(refNode_probs, refNode_rule_idx, [7], 0, valid_sample_idxs,
                                       refNode_nums, query_rel_flatten, probs, refEdges, preds, mode='Train')
        check("update_outputs train returns None", result is None)
        check("update_outputs valid sample", valid_sample_idxs == [0], f"got {valid_sample_idxs}")
        check("update_outputs refNode nums", refNode_nums == [1], f"got {refNode_nums}")
        check("update_outputs query rel", query_rel_flatten == [7], f"got {query_rel_flatten}")
        check("update_outputs refEdges count", len(refEdges) == 2, f"got {len(refEdges)}")
    except Exception as exc:
        skip_checks(11, f"TEKG_fast_ver test raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 4/5] TEKG connectivity and event probabilities")
    try:
        option = make_option(flag_interval=True)
        option.prob_selection_for_training = 'max'
        graph = TEKG(option, data={'num_rel': 4})
        nodes = np.array([
            [0, 0, 1, 2000, 2000],
            [1, 1, 2, 2001, 2001],
            [2, 2, 0, 1999, 1999],
        ])
        aug_nodes, aug_idx = graph._build_aug_nodes(nodes, num_entity=6)
        check("aug nodes shape", aug_nodes.shape == (6, 5), f"got {aug_nodes.shape}")
        check("aug idx values", aug_idx.tolist() == [0, 1, 2, 3, 4, 5], f"got {aug_idx.tolist()}")
        conn_rel = graph._build_connectivity_rel(aug_nodes, aug_idx, 6, 4)
        check("connectivity rel keys", set(conn_rel.keys()) == {0, 1, 2, 3})
        check("connectivity rel nonempty", len(conn_rel[0][0]) >= 1)
        conn_tr = graph._build_connectivity_TR(aug_nodes, aug_idx, 6, 4)
        check("connectivity TR keys", set(conn_tr.keys()) == {0, 1, 2, 3})
        check("connectivity unknown shape", conn_tr[0][2] == [6, 6])
        check("connectivity has temporal edge", any(len(conn_tr[i][0]) > 0 for i in [1, 2, 3]))

        TEKG_nodes_aug = aug_nodes
        TEKG_nodes_idx_aug = aug_idx
        edge_for_walk = str((0, 0, 2000, 2000, 1))
        walk_res = {'0': {edge_for_walk: {'0': [[0.2], [0.5]], '1': [], '2': [], '3': []}}, '1': {}}
        probs = [[[0.0 for _ in range(6)] for _ in range(8)]]
        flag_valid, ref_event_idx = graph._update_event_probabilities(
            probs, 0, 10, walk_res, 6, TEKG_nodes_aug, TEKG_nodes_idx_aug,
            mode='Train', stage=None, final_preds=None, extra_data=None
        )
        check("event update valid", flag_valid == 1)
        check("event update train no ref idx", ref_event_idx == {})
        check("event update writes max prob", abs(probs[0][2][3] - 0.5) < 1e-8, f"got {probs[0][2][3]}")
    except Exception as exc:
        skip_checks(12, f"TEKG connectivity test raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 5/5] Learner TensorFlow graph-building methods")
    try:
        learner = object.__new__(Learner)
        learner.norm = True
        learner.thr = 1e-20
        learner.prob_scaling_factor = [0.1, 0.5]
        learner.connectivity_TR = {
            0: tf.SparseTensor(indices=[[0, 0], [1, 1], [2, 2]], values=[1.0, 1.0, 1.0], dense_shape=[3, 3])
        }
        state_vec = tf.constant([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], dtype=tf.float32)
        query_time_dist = tf.constant([[0.4, 0.2, 0.6], [0.5, 0.1, 0.3]], dtype=tf.float32)
        target_entities = tf.constant([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=tf.float32)
        pred = learner._time_prediction(state_vec, query_time_dist, target_entities)
        check("time_prediction output not None", pred is not None)

        trans_mat = {
            0: tf.SparseTensor(indices=[[0, 0], [1, 1], [2, 2]], values=[1.0, 1.0, 1.0], dense_shape=[3, 3]),
            1: tf.SparseTensor(indices=[[0, 1], [1, 2], [2, 0]], values=[1.0, 1.0, 1.0], dense_shape=[3, 3]),
        }
        attn = [tf.constant([[0.7], [0.7]], dtype=tf.float32), tf.constant([[0.3], [0.3]], dtype=tf.float32)]
        selected = learner._selection_block(state_vec, attn, trans_mat, [0, 1])
        check("selection_block output not None", selected is not None)
        scaled_small = learner._scale_final_prob(tf.constant([[0.05], [0.08]], dtype=tf.float32))
        scaled_large = learner._scale_final_prob(tf.constant([[0.20], [0.08]], dtype=tf.float32))
        check("scale_final_prob small output not None", scaled_small is not None)
        check("scale_final_prob large output not None", scaled_large is not None)
        check("tensorflow tensors expose shapes", pred.shape.as_list() == [2, 1] and selected.shape.as_list() == [2, 3])
        check("scaled tensors expose shapes", scaled_small.shape.as_list() == [2, 1] and scaled_large.shape.as_list() == [2, 1])
    except Exception as exc:
        skip_checks(7, f"Learner TensorFlow graph test raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
        raise SystemExit(1)
