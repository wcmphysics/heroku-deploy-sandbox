# Deploying a model

The next steps

1. Check if API is working locally with
- ``` python main.py```
- ``` gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app```
2. Deploy to Heroku
3. Enjoy the result


The ```requirements.txt``` should not contain the development environment (e.g. jupyter) and as few libraries as possible, because Memory for the App is very limited.

## General Tips for deployment

### Prior to deployment

- Make sure you clean up, process and strip down the data as much as you can first.
- You have limited space, so don't waste space.
- Test your app locally thoroughly first. Use an environment that uses the "requirements.txt" for this so you are using the same packages as will be used.

Make sure you have these three files in the repo:

1. runtime.txt -> This is the runtime environment for the app, defines which python version is used. Make sure that the service offers your python version!
2. requirements.txt
3. Procfile -> The Procfile defines which process is used for the app.

## Deployment to Heroku
The General steps are:

1. First you need to login or create an Account on Heroku
2. click on New -> New Apps (on the right side)
3. Pick an unique name, this is gonna be reflected in your apps URL
4. Select europe
5. Use the github connection (allow and set this up if required)
6. Pick your repository
7. Select the branch (you should have a specific repository for the deployment or, at least, a specific branch)
8. Hit deploy and wait for the build
9. Hit view and
10. Enjoy!
