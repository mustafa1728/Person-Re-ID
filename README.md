# Person Re Identification


Experiments on some existing Re-ID methods on a different dataset with qualitative and quantitaive analysescof their performance and proposals of methods to improve the results further.

### Description

A detailed description of the problem statement and scope of the work can be found [here](./assets/report.pdf).

### Baselines

- AlignedReID [[paper](https://arxiv.org/abs/1711.08184)][[paper](https://www.sciencedirect.com/science/article/pii/S0031320319302031?casa_token=i87oHbS1tg0AAAAA:91ksksd2huP72e0nzDKFLtnJU3hAYHx4mm-dpBxwdjEYJdHjf8xFIXwSXrVIaW2Wn_Ociqy8xA)]

- LA Transformer [[paper](https://arxiv.org/abs/2106.03720)]

- Centroids ReID [[paper](https://arxiv.org/abs/2104.13643)]

- TransReID [[paper](https://arxiv.org/abs/2102.04378)]


<p align="center">
  <img src="assets/base_preds.png" width="800"/>
  <br>The top 5 closest images in the gallery for a few querry images with the baselines
</p>


<p align="center">
  <img src="assets/LA_Transformer_baseline.png" width="400"/>
  <img src="assets/AlignedReID_baseline.png" width="400"/>
  <br>t-SNE plots
</p>

For further details and description of the proposed methods, please refer to the report [here](./assets/report.pdf).
