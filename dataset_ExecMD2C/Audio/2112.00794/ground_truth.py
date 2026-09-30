"""
Self-contained benchmark answer key for the core DFTS2 packet transmission
and error-concealment algorithms.

This file consolidates the repository's core model-adjacent components from:
- models/packetModel.py
- models/quantizer.py
- errConceal/altec.py
- errConceal/caltec.py

Fitting scripts, data ingestion, experiment launchers, configuration generation,
and deep model-splitting wrappers are intentionally excluded from this
reproduction benchmark.
"""

from timeit import default_timer as timer

import numpy as np


class PacketModel(object):
    def __init__(self, **kwargs):
        allowed_keys = {'rows_per_packet', 'data_tensor', 'packet_seq', 'data_shape'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        if 'packet_seq' not in kwargs.keys():
            self.data_shape = self.data_tensor.shape
            self.packet_seq = self.dataToPacket()
        else:
            if self.data_shape[1] % self.rows_per_packet == 0:
                self.numZeros = 0
            else:
                self.numZeros = self.rows_per_packet - (self.data_shape[1] % self.rows_per_packet)
            self.data_tensor = self.packetToData()

    def dataToPacket(self):
        self.numZeros = 0
        if self.data_shape[1] % self.rows_per_packet == 0:
            self.packet_seq = np.reshape(
                self.data_tensor,
                (self.data_shape[0], -1, self.rows_per_packet, self.data_shape[2], self.data_shape[3]),
            )
            return self.packet_seq

        self.numZeros = self.rows_per_packet - (self.data_shape[1] % self.rows_per_packet)
        zeros = np.zeros((self.data_shape[0], self.numZeros, self.data_shape[2], self.data_shape[3]))
        packets = np.concatenate((self.data_tensor, zeros), axis=1)
        self.packet_seq = np.reshape(
            packets,
            (self.data_shape[0], -1, self.rows_per_packet, self.data_shape[2], self.data_shape[3]),
        )
        return self.packet_seq

    def packetToData(self):
        if self.numZeros == 0:
            self.data_tensor = np.reshape(self.packet_seq, self.data_shape)
            return self.data_tensor

        packetSeq = np.reshape(self.packet_seq, (self.data_shape[0], -1, self.data_shape[2], self.data_shape[3]))
        index = -1 * self.numZeros
        self.data_tensor = packetSeq[:, :index, :, :]
        return self.data_tensor


class QLayer(object):
    def __init__(self, nBits):
        super(QLayer, self).__init__()
        self.nBits = nBits

    def bitQuantizer(self, data):
        start_time = timer()
        self.max = np.max(data)
        self.min = np.min(data)
        np.seterr(divide='ignore', invalid='ignore')
        self.quanData = np.round(((data - self.min) / (self.max - self.min)) * ((2**self.nBits) - 1))
        total_time = timer() - start_time
        print(f"Bit quantization complete in {total_time:.3f}s")

    def inverseQuantizer(self):
        self.quanData = (self.quanData * (self.max - self.min) / ((2**self.nBits) - 1)) + self.min
        return self.quanData


def fn_compute_altec_weights_pkts(pkt_model):
    """
    Compute ALTeC weights with a DFTS packet model.
    This is the modified ALTeC algorithm to work with packets with multiple
    consecutive rows of features per packet (rowsPerPacket > 1).
    Based on ALTeC_train().
    Adapted from canonical implementation developed by Lior Bragilevsky (SFU
    Multimedia Lab).
    """
    num_examples = np.shape(pkt_model.packet_seq)[0]
    num_pkts = np.shape(pkt_model.packet_seq)[1]
    rowsPerPacket = np.shape(pkt_model.packet_seq)[2]
    channel_width = np.shape(pkt_model.packet_seq)[3]
    num_channels = np.shape(pkt_model.packet_seq)[4]

    train_example = PacketModel(rows_per_packet=rowsPerPacket, data_tensor=np.copy(pkt_model.data_tensor))
    channel_weights = np.zeros([num_examples, num_channels + 1, num_channels], dtype=np.float32)
    # Process each example in the packetized tensor.
    for i_example in range(num_examples):
        for i_c in range(num_channels):
            # Loop through all channels in tensor
            w_pkts = np.zeros([num_channels + 1, num_pkts], dtype=np.float32)
            for i_pkt in range(num_pkts):
                # Initialize for each packet in a channel
                X_i_pkt = np.zeros([num_channels + 1, channel_width * rowsPerPacket], dtype=np.float32)
                # fill in X_i_pkt
                i_k_channel = 0
                for i_k in range(num_channels):
                    if i_k == i_c:
                        # Co-located packet in the same channel is excluded.
                        # print(f'Processing packet {i_pkt} in channel {i_c}. Excluding channel {i_k} in colocated packets.')
                        continue
                    else:
                        # print(f'Colocated packet in other channel {i_k}. Saved in channel {i_k_channel}')
                        # Co-located packets in other channels.
                        X_i_pkt[i_k_channel, :] = np.reshape(
                            train_example.packet_seq[i_example, i_pkt, :, :, i_k],
                            (channel_width * rowsPerPacket),
                        )
                        i_k_channel += 1
                if i_pkt > 0:
                    # Spatial neighbor below in-channel.
                    X_i_pkt[-2, :] = np.reshape(
                        train_example.packet_seq[i_example, i_pkt - 1, :, :, i_c],
                        (channel_width * rowsPerPacket,),
                    )
                if i_pkt < num_pkts - 1:
                    # Spatial neighbor above in-channel.
                    X_i_pkt[-1, :] = np.reshape(
                        train_example.packet_seq[i_example, i_pkt + 1, :, :, i_c],
                        (channel_width * rowsPerPacket,),
                    )
                # Compute weights.
                pseudo = np.linalg.pinv(X_i_pkt).transpose()
                w_pkts[:, i_pkt] = pseudo.dot(
                    np.reshape(train_example.packet_seq[i_example, i_pkt, :, :, i_c], (channel_width * rowsPerPacket))
                )
            avg_w_pkts = np.mean(w_pkts, axis=1)
            # Accumulate weights for a given channel.
            channel_weights[i_example, :, i_c] = avg_w_pkts
    return np.mean(channel_weights, axis=0)  # Average weights by the number of examples in a batch.


def fn_complete_tensor_altec_pkts_star(corrupted_tensor, rowsPerPacket, altec_weights_pkt, loss_map):
    """
    Complete a corrupted tensor in a DFTS packet model using ALTeC weights.
    Use a loss map to figure out which packet needs to be repaired.
    Adapted from canonical implementation developed by Lior Bragilevsky (SFU
    Multimedia Lab).
    Compatible with weights produced by fn_compute_altec_weights_pkts.
    adapted tuesday jan 20. for star
    """
    repaired_pkt_model = PacketModel(rows_per_packet=rowsPerPacket, data_tensor=np.copy(corrupted_tensor))
    num_examples = np.shape(repaired_pkt_model.packet_seq)[0]
    num_pkts = np.shape(repaired_pkt_model.packet_seq)[1]
    channel_width = np.shape(repaired_pkt_model.packet_seq)[3]
    num_channels = np.shape(repaired_pkt_model.packet_seq)[4]

    for i_example in range(num_examples):
        # Process all items in packetized tensor.
        item_loss_map = loss_map[i_example, :, :]

        for i_c in range(num_channels):
            # Loop through all channels in tensor
            for i_pkt in range(num_pkts):
                if item_loss_map[i_pkt, i_c] == False:
                    # If packet was lost, repair it.
                    # Initialize for each packet in a channel
                    X_i_pkt = np.zeros([num_channels + 1, channel_width * rowsPerPacket], dtype=np.float32)
                    # fill in X_i_pkt
                    i_k_channel = 0
                    for i_k in range(num_channels):
                        if i_k == i_c:
                            # Co-located packet in the same channel is excluded.
                            continue
                        else:
                            # Co-located packets in other channels.
                            X_i_pkt[i_k_channel, :] = np.reshape(
                                repaired_pkt_model.packet_seq[i_example, i_pkt, :, :, i_k],
                                (channel_width * rowsPerPacket),
                            )
                            i_k_channel += 1
                    if i_pkt > 0:
                        X_i_pkt[-2, :] = np.reshape(
                            repaired_pkt_model.packet_seq[i_example, i_pkt - 1, :, :, i_c],
                            (channel_width * rowsPerPacket,),
                        )
                    if i_pkt < num_pkts - 1:
                        X_i_pkt[-1, :] = np.reshape(
                            repaired_pkt_model.packet_seq[i_example, i_pkt + 1, :, :, i_c],
                            (channel_width * rowsPerPacket,),
                        )

                    estimated_pkt_reshaped = np.dot(X_i_pkt.T, altec_weights_pkt[:, i_c])
                    estimated_pkt = np.reshape(estimated_pkt_reshaped, (rowsPerPacket, channel_width))
                    repaired_pkt_model.packet_seq[i_example, i_pkt, :, :, i_c] = estimated_pkt

    repaired_tensor = np.copy(repaired_pkt_model.data_tensor)
    return repaired_tensor


def find_nearest_index(array, value):
    idx = np.searchsorted(array, value, side='left')
    if idx > 0 and (idx == len(array)):
        return idx - 1
    else:
        return idx


def fn_caltec(lossMatrix, pkt_obj, item_index):
    """
    Implementation of the CALTeC packet-repair algorithm for one item in a
    packetized DFTS feature tensor.
    """
    # Extract information from the packet model object.
    num_channels = pkt_obj.packet_seq.shape[4]
    channel_width = pkt_obj.packet_seq.shape[3]
    rowsPerPacket = pkt_obj.packet_seq.shape[2]
    num_pkts_per_channel = pkt_obj.packet_seq.shape[1]
    lost_map = lossMatrix[item_index, :, :]

    if np.size(lost_map) - np.count_nonzero(lost_map) == 0:
        print(f"No packets were lost for item {item_index}.")
        return pkt_obj

    for i_c in range(num_channels):
        # Check if the whole channel is lost.
        if np.all(lost_map[:, i_c] == False):
            print(f"All packets lost in channel {i_c}. Could not repair this channel.")
            continue

        for i_pkt in range(num_pkts_per_channel):
            # Check if the packet is lost.
            if lost_map[i_pkt, i_c] == False:
                # Check if other channels have co-located packets.
                existing_colocated_pkts_list = np.where(lost_map[i_pkt, :] == True)[0]
                if len(existing_colocated_pkts_list) == 0:
                    print(f"No co-located packets found for item {item_index}, packet {i_pkt}, channel {i_c}.")
                    continue

                # Search for existing packets in the damaged channel.
                existing_pkts_in_channel_list = np.where(lost_map[:, i_c] == True)[0]
                nearest_index = find_nearest_index(existing_pkts_in_channel_list, i_pkt)
                for test_pkt_id in range(nearest_index - 1, -num_pkts_per_channel, -1):
                    # Select nearest packet in damaged channel.
                    nearest_neighbor_in_channel_idx = existing_pkts_in_channel_list[test_pkt_id]
                    # Select co-located packets at the nearest packet in damaged channel.
                    existing_colocated_pkts_neighbor_list = np.where(lost_map[nearest_neighbor_in_channel_idx, :] == True)[0]
                    # Exclude the damaged channel.
                    existing_colocated_pkts_neighbor_list = np.delete(
                        existing_colocated_pkts_neighbor_list,
                        np.where(existing_colocated_pkts_neighbor_list == i_c),
                    )

                    # Identify channels that have the packet at the lost position
                    # and also at the selected nearest position.
                    candidate_channels = np.intersect1d(
                        existing_colocated_pkts_list,
                        existing_colocated_pkts_neighbor_list,
                    )
                    if len(candidate_channels) > 0:
                        break

                if len(candidate_channels) == 0:
                    print(f"No co-located packets found for item {item_index}, packet {i_pkt}, channel {i_c}.")
                    continue

                candidate_channels = np.concatenate(([i_c], candidate_channels), axis=0)
                nearest_neighbor_pkt = pkt_obj.packet_seq[
                    item_index,
                    nearest_neighbor_in_channel_idx,
                    :,
                    :,
                    i_c,
                ]
                corrcoef_matrix = np.corrcoef(
                    np.reshape(
                        pkt_obj.packet_seq[
                            item_index,
                            nearest_neighbor_in_channel_idx,
                            :,
                            :,
                            candidate_channels,
                        ],
                        (rowsPerPacket * channel_width, -1),
                    ).T
                )
                temp = np.sort(corrcoef_matrix[0, :])
                high_corr = temp[-2]
                temp = np.where(corrcoef_matrix[0, :] == high_corr)[0][-1]
                highest_corr_channel_idx = candidate_channels[temp]

                # Estimate the lost packet from a high-correlation co-located
                # packet and a linear luminance transform.
                pkt_from_other_channel = pkt_obj.packet_seq[
                    item_index,
                    i_pkt,
                    :,
                    :,
                    highest_corr_channel_idx,
                ]
                neighbor_out_channel = pkt_obj.packet_seq[
                    item_index,
                    nearest_neighbor_in_channel_idx,
                    :,
                    :,
                    highest_corr_channel_idx,
                ]
                vec_in_channel = np.reshape(nearest_neighbor_pkt, (channel_width * rowsPerPacket))
                vec_out_channel = np.reshape(neighbor_out_channel, (channel_width * rowsPerPacket))
                lumi_transf = np.polyfit(vec_out_channel, vec_in_channel, 1)
                lin_func = np.poly1d(lumi_transf)
                est_pkt = lin_func(np.reshape(pkt_from_other_channel, (channel_width * rowsPerPacket)))
                pkt_obj.packet_seq[item_index, i_pkt, :, :, i_c] = np.reshape(est_pkt, (rowsPerPacket, channel_width))

    return pkt_obj


if __name__ == "__main__":
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

    print("Running DFTS2 benchmark checks...")

    try:
        data = np.arange(1 * 5 * 3 * 2, dtype=np.float32).reshape(1, 5, 3, 2)
        pkt = PacketModel(rows_per_packet=2, data_tensor=data)
        restored = PacketModel(rows_per_packet=2, packet_seq=pkt.packet_seq, data_shape=data.shape)
        check("packet padded shape", pkt.packet_seq.shape == (1, 3, 2, 3, 2), str(pkt.packet_seq.shape))
        check("packet padding count", pkt.numZeros == 1, str(pkt.numZeros))
        check("packet padding values", np.allclose(pkt.packet_seq[:, -1, -1, :, :], 0.0))
        check("packet round trip padded", np.array_equal(restored.data_tensor, data))

        data_even = np.arange(2 * 4 * 2 * 1, dtype=np.float32).reshape(2, 4, 2, 1)
        pkt_even = PacketModel(rows_per_packet=2, data_tensor=data_even)
        restored_even = PacketModel(rows_per_packet=2, packet_seq=pkt_even.packet_seq, data_shape=data_even.shape)
        check("packet exact shape", pkt_even.packet_seq.shape == (2, 2, 2, 2, 1), str(pkt_even.packet_seq.shape))
        check("packet exact no padding", pkt_even.numZeros == 0, str(pkt_even.numZeros))
        check("packet round trip exact", np.array_equal(restored_even.data_tensor, data_even))
    except Exception as exc:
        skip_checks(7, f"packet model checks raised {type(exc).__name__}: {exc}")

    try:
        source = np.linspace(-3.0, 5.0, 17, dtype=np.float32).reshape(1, 17, 1, 1)
        quantizer = QLayer(4)
        quantizer.bitQuantizer(source)
        qdata = quantizer.quanData.copy()
        restored_quant = quantizer.inverseQuantizer()
        step = (float(np.max(source)) - float(np.min(source))) / ((2**4) - 1)
        check("quantizer min max", quantizer.min == np.min(source) and quantizer.max == np.max(source))
        check("quantizer integer levels", np.all(qdata == np.round(qdata)))
        check("quantizer level range", np.min(qdata) >= 0 and np.max(qdata) <= 15)
        check("quantizer shape preserved", restored_quant.shape == source.shape, str(restored_quant.shape))
        check("quantizer inverse tolerance", np.max(np.abs(restored_quant - source)) <= step / 2 + 1e-6)
    except Exception as exc:
        skip_checks(5, f"quantizer checks raised {type(exc).__name__}: {exc}")

    try:
        train_tensor = np.zeros((1, 3, 4, 3), dtype=np.float32)
        train_tensor[0, :, :, 0] = np.array([[1, 2, 3, 4], [5, 7, 9, 11], [2, 4, 6, 8]], dtype=np.float32)
        train_tensor[0, :, :, 1] = np.array([[2, 1, 0, 1], [1, 3, 5, 7], [8, 6, 4, 2]], dtype=np.float32)
        train_tensor[0, :, :, 2] = np.array([[3, 5, 7, 9], [9, 7, 5, 3], [1, 0, 1, 0]], dtype=np.float32)
        train_pkt = PacketModel(rows_per_packet=1, data_tensor=train_tensor)
        weights = fn_compute_altec_weights_pkts(train_pkt)
        check("altec weights shape", weights.shape == (4, 3), str(weights.shape))
        check("altec weights finite", np.all(np.isfinite(weights)))

        corrupted = train_tensor.copy()
        corrupted[0, 1, :, 0] = 0.0
        loss_map = np.ones((1, 3, 3), dtype=bool)
        loss_map[0, 1, 0] = False
        repaired = fn_complete_tensor_altec_pkts_star(corrupted, 1, weights, loss_map)
        check("altec repair shape", repaired.shape == corrupted.shape, str(repaired.shape))
        check("altec repair finite", np.all(np.isfinite(repaired)))
        check("altec observed packet stable", np.allclose(repaired[0, 0, :, 1], corrupted[0, 0, :, 1]))
        check("altec repaired packet changed", not np.allclose(repaired[0, 1, :, 0], 0.0))
    except Exception as exc:
        skip_checks(6, f"ALTeC checks raised {type(exc).__name__}: {exc}")

    try:
        array = np.array([0, 2, 5, 7])
        check("nearest lower boundary", find_nearest_index(array, -1) == 0)
        check("nearest insertion point", find_nearest_index(array, 3) == 2)
        check("nearest upper clamp", find_nearest_index(array, 99) == 3)

        packet_seq = np.zeros((1, 3, 1, 4, 3), dtype=np.float32)
        packet_seq[0, 0, 0, :, 0] = np.array([1, 2, 3, 4], dtype=np.float32)
        packet_seq[0, 2, 0, :, 0] = np.array([10, 20, 30, 40], dtype=np.float32)
        packet_seq[0, 1, 0, :, 1] = np.array([201, 401, 601, 801], dtype=np.float32)
        packet_seq[0, 2, 0, :, 1] = np.array([21, 41, 61, 81], dtype=np.float32)
        packet_seq[0, 0, 0, :, 1] = np.array([9, 8, 7, 6], dtype=np.float32)
        packet_seq[0, :, 0, :, 2] = np.array(
            [[4, 3, 2, 1], [2, 3, 2, 3], [1, 4, 2, 5]],
            dtype=np.float32,
        )
        loss_matrix = np.ones((1, 3, 3), dtype=bool)
        loss_matrix[0, 1, 0] = False
        caltec_pkt = PacketModel(rows_per_packet=1, packet_seq=packet_seq.copy(), data_shape=(1, 3, 4, 3))
        repaired_pkt = fn_caltec(loss_matrix, caltec_pkt, 0)
        expected = np.array([-191, -391, -591, -791], dtype=np.float32)
        check("caltec repaired packet", np.allclose(repaired_pkt.packet_seq[0, 1, 0, :, 0], expected, atol=1e-3))
        check("caltec observed channel stable", np.allclose(repaired_pkt.packet_seq[0, 1, 0, :, 1], packet_seq[0, 1, 0, :, 1]))
        check("caltec repair finite", np.all(np.isfinite(repaired_pkt.packet_seq)))
        clean_pkt = PacketModel(rows_per_packet=1, packet_seq=packet_seq.copy(), data_shape=(1, 3, 4, 3))
        returned = fn_caltec(np.ones((1, 3, 3), dtype=bool), clean_pkt, 0)
        check("caltec no loss identity", np.allclose(returned.packet_seq, packet_seq))
    except Exception as exc:
        skip_checks(7, f"CALTeC checks raised {type(exc).__name__}: {exc}")

    total = passed + failed
    print(f"Checks passed: {passed}/{total}")
    if failed:
        raise SystemExit(1)
