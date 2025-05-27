# Homework 1

Task description:
> Based on the CNN landcover detection notebook, replace the vanilla CNN by a ResNet (you can take that from the internet, no need to write from scratch). Use a small ResNet which you can still train for at least a few steps. Then add code to evaluate your model (the CNN or/and the ResNet) by 1) calculating the kappa statistics, and 2) show an ROC curve which you can generate by varying the threshold for class assignment at the output of the CNN.

## Solution

We implement a `MinimalResNet` that uses residual blocks with skip connections and batch normalization and a total of 2 residual layers. We find that for the given reduced-size dataset, both CNN and ResNet with similar training parameters can both accurately classify the distinct observations.

However, ResNet is even faster and would require less epochs. Overall, the distinction of some class labels is more difficult than others (road vs. building, trees vs. grassland), while other labels are particularly easy to classify, such as water due to the NIR channel information. We present findings in the confusion matrices reported by both models. Cohen's kappa for the CNN is 0.9460, and for the ResNet model 0.9700. ROC curves do not yield as much insight, as both models are very accurate already.