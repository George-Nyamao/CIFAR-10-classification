pipeline {
	agent { docker { image 'python:3.7.6' } }
	stages {
		stage ('Clone Repository'){
		/* Cloning the repository for our workspace*/
			steps {
				checkout scm
			}
		}
		stage ('Build Image') {
			steps {
				withEnv(["HOME=$env.WORKSPACE"]){
					sh 'docker build -t cifar_10_classifier:v2 .'
				}
			}
		}
		stage ('Run Container') {
			steps {
				echo 'Running container..'
				withEnv(["HOME=$env.WORKSPACE"]){
					sh 'docker run -d -p 8008:8080 --name cifar10 cifar_10_classifier'
				}
			}
		}
		stage ('Testing') {
			steps {
				echo 'Testing..'
			}
		}
	}
}