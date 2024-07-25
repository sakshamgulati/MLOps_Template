#create a docker image with this command, replace sakshamgulati with your dockerhub username
# docker build -t sakshamgulati/mlops_template .

#create a step function with this command, you need to manually trigger the step function for the very first run
python infer.py --environment=pypi step-functions create

#run the step function with this command for execution
python infer.py --environment=pypi run --with batch

#run the step function with this command for execution
{
    "Parameters" : "{}"
}