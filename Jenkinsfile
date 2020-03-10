pipeline {
	agent { docker {image 'python:3.7.6'}}
	stages {
		stage ('Clone Repository'){
			steps {
				checkout scm
			}
		}
		stage ('Build Image') {
			steps {
				sh 'docker build --tag=cifar_10_classifier .'
			}
		}
		stage ('Run Container') {
			steps {
				echo 'Running container..'
				sh 'docker run -d -p 8008:8080 cifar10 cifar_10_classifier'
			}
		}
		stage ('Open Browser') {
			steps {
				echo 'Testing..'
                sh 'google-chrome http://127.0.0.1:8008'
			}
		}
	}
}