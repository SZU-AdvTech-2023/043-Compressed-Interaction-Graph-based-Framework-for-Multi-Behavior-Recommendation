# Reproduction Report of the Compressed Interaction Graph based Framework for Muti-behavior Recommendation

## Environment
The codes of CIGF are implemented and tested under the following development environment:
* python=3.6.12
* tensorflow=1.14.0
* numpy=1.16.0
* scipy=1.5.2

## Datasets
Utilized three datasets to evaluate CIGF: <i>Beibei, Tmall, </i>and <i>JD</i>. The <i>purchase</i> behavior is taken as the target behavior for all datasets. The last target behavior for the test users are left out to compose the testing set. 

## Just Run It！

* JD
```
python base_framework_origin_final.py --data JD 

* Beibei
```
python base_framework_origin_final.py --data beibei 
```
* Tmall
```
python base_framework_origin_final.py --data tmall --gnn_layer 4
```


## Citation
 cite:
```
@inproceedings{cigf,
  title={Compressed Interaction Graph based Framework for Multi-behavior Recommendation},
  author={Guo, Wei and Meng, Chang and Yuan, Enming and He, Zhicheng and Guo, Huifeng and Zhang, Yingxue and Chen, Bo and Hu, Yaochen and Tang, Ruiming and Li, Xiu and others},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={960--970},
  year={2023}
}
```
