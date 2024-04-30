This project serves as a guide to implementing end to end ML Lifecycle using Continuous Integration and Continuous Deployment (CICD)   when developing an end-to-end ML product. It encompasses several key aspects:

1. **Development and Experimentation:** Our journey begins with rapid prototyping in Jupyter notebooks, followed by experimentation with our algorithms using Weights and Biases for tracking iterations. Internal modules have been crafted to facilitate data loading, preprocessing, model training, and model deployment. While these modules are tailored to the specific project, you'll find a simplified example in the repository to illustrate my approach. Currently, the project relies on using Diabetes example, while more complex examples are work under progress.

2. **Tracking and Environment**  - I have used Pipenv to create virtual environments to ensure experiments are trackable. 
3. **Production:** Metaflow has been leveraged to create Directed Acyclic Graphs (DAGs) designed for deployment on cloud providers like AWS and GCP. It's worth noting that the default deployment method is currently local; however, any changes to this approach can be easily configured in the `.metaflow` configuration files. Note that in the current strategy, we are storing our model on Model Registry (Weights and bias) and later inferencing on the best model from the same registry

4. **Continuous Integration/Continuous Deployment:** Our CICD pipeline relies on GitHub Actions and Metaflow scripts, making it straightforward to train models on cost-free Ubuntu machines provided by GitHub. Once your model is trained, you can seamlessly publish it in your model repository, managed by Weights and Biases.

Should you choose to train a different, improved model, you can utilize the same scripts to train and publish it within the same model repository. However, when transitioning to deploying the model in a production environment, it's essential to pass multiple unit tests and adhere to linting standards, all of which are powered by Pytest.

For more information-https://www.notion.so/Machine-Learning-Ops-3983660707ff41fdaa37717bc8153140?pvs=4

**Next Steps**
1. Replacing Diabetes with Time-Series Model ensuring model retraining capability
2. Incorporating Model Retraining and model replacing with CICD
3. Capturing Model/Feature Drift using Evidently AI
4. Shifting Model to AWS
5. Dockerization and shifting model to K8s

**Inspiration-**
<br>Idea inspired by <a href='https://github.com/simonprudhomme'> Simon </a>
<img width="443" alt="image" src="https://github.com/sakshamgulati/MLOps_Template/assets/16202917/c175e03e-c753-474f-b6a5-17b792b297e2">
