# Evaluating SAM against SGD 
To reproduce results regarding SAM's performance against SGD including Figure 5, 
1. Download `Model_SAM_EfficientNetb0.ipynb`
2. Upload file to Google Drive
3. Select GPU T4 as a ressource
4. Run the entire file. Results are clearly output throughout the file.

## Acknowledgments
The original authors of SAM
```bash
@inproceedings{foret2021sharpnessaware,
  title={Sharpness-aware Minimization for Efficiently Improving Generalization},
  author={Pierre Foret and Ariel Kleiner and Hossein Mobahi and Behnam Neyshabur},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=6Tm1mposlrM}
}```
- The SAM algorithm implemented is used from the paper [SHARPNESS-AWARE MINIMIZATION FOR EFFICIENTLY
IMPROVING GENERALIZATION]([https://link.com](https://arxiv.org/abs/2010.01412)) 
- The code used in this repository was heavily aided from the repository [Sharpness-Aware Minimization for Efficiently Improving Generalization ~ in Pytorch ~]([https://link.com](https://github.com/davda54/sam))
