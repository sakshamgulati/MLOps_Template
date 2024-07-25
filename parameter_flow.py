from metaflow import FlowSpec, Parameter, step, secrets,conda_base, batch, retry,IncludeFile
import os

# @conda_base(python='3.10', packages={'python-dotenv':'0.21.1'})
class ParameterFlow(FlowSpec):
    alpha = Parameter('alpha',
                      help='Learning rate',
                      default=0.01)
    variant = IncludeFile(
        'variant',
        is_text=False,
        # required=True,
        default="conf/mlops.yaml")
    
    @secrets(sources=['wandb_api_key'])
    @batch(memory=8000, cpu=1)
    @retry    
    @step
    def start(self):
        import json
        print('alpha is %f' % self.alpha)
        self.conf = json.loads(self.variant)
        print(self.conf['model_name'])
        self.next(self.end)

    @step
    def end(self):
        print('alpha is still %f' % self.alpha)

if __name__ == '__main__':
    ParameterFlow()