'''
The code is used to merge different results to obtain a new result.
For example, we can utilize 'Flickr30K_DynamicTopK_P_3_sim.npy'(HGMN w/o S) and 'Flickr30K_DynamicTopK_S_3_sim.npy'(HGMN w/o P)
to obtain 'Flickr30K_DynamicTopK_S+P_3_sim.npy'(HGMN).

'''

import numpy as np
def i2t(im_len, sims, npts=None, return_ranks=False):

    npts = im_len
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(im_len, sims, npts=None, return_ranks=False):

    npts = im_len
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


if __name__ == '__main__':
    # reading  the similarity matrix
    sim_1 = np.load('./data/TopK_V2T_NTN.npy')
    sim_2 = np.load('./data/TopK_V2T_P.npy')

    ima_len, caps_len = sim_1.shape
    Com_results = []
    isfold5 = False
    if not isfold5:
        for i in range(101):
            alpha = i/100.0
            print(alpha)
            sims = alpha*sim_1 + (1.0-alpha)*sim_2

            r, rt = i2t(ima_len, sims, return_ranks=True)
            ri, rti = t2i(ima_len, sims, return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            Com_results.append((rsum, r[0], r[1], r[2], ri[0], ri[1], ri[2], i))
           # Com_results.append(rsum)

    # # MSCOCO 5K
    else:
        results = []
        for i in range(5):
            sim_shard = (sim_1[i] + sim_2[i]) / 2
            r, rt0 = i2t(ima_len, sim_shard, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(ima_len, sim_shard, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    # with open('result.txt', 'w') as f:
    #     for i in range(len(Com_results)):
    #         f.write(str(i/100.0)+','+ str(Com_results[i]) + '\r')


    print(Com_results)
    print(max(Com_results))

