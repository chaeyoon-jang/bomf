# **Model Fusion through Bayesian Optimization in Language Model Fine-Tuning**

This repository provides the codebase for our NeurIPS 2024 paper:  
**[Model Fusion through Bayesian Optimization in Language Model Fine-Tuning](https://arxiv.org/abs/2411.06710)**.

In this work, we propose an innovative **model fusion technique** that leverages **multi-objective Bayesian optimization** to optimize desired metrics and loss simultaneously. We also introduce a two-stage hyperparameter selection framework that integrates Bayesian optimization into the fine-tuning process.

---

## **Setup**

1. Create and activate the environment:  
   ```bash
   conda create -n bo-fusion
   conda activate bo-fusion
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

---

## **Experiments**

Run the following commands for different experimental steps:

1. **Hyperparameter Optimization**  
   Find the best hyperparameters:  
   ```bash
   bash hpbo_{data_type}.sh
   ```

2. **Model Fusion**  
   Discover the best combination of checkpoints:  
   ```bash
   bash bomf_{data_type}.sh
   ```
   
Replace the placeholders as follows:  
- `data_type`: `squad`, `glue`, `samsum`, etc.

---

## **Citation**

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{jangmodel,
  title={Model Fusion through Bayesian Optimization in Language Model Fine-Tuning},
  author={Jang, Chaeyun and Lee, Hyungi and Kim, Jungtaek and Lee, Juho},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```