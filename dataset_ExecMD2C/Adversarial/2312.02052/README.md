# DUCK: Distance-based Unlearning via Centroid Kinematics (2312.02052)

## Project source

- **Project ID:** `2312.02052`
- **Paper:** DUCK: Distance-based Unlearning via Centroid Kinematics
- **PDF:** https://arxiv.org/pdf/2312.02052v2.pdf
- **GitHub:** https://github.com/ocram17/duck

## Reproduction task

DUCK removes selected training knowledge by moving embeddings away from class centroids. In `template.py`, complete `DUCK.pairwise_cos_dist` and `DUCK.run` to compute the pairwise cosine distance matrix used to match forget embeddings to centroids and execute DUCK's centroid-guided unlearning procedure.
