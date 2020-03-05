pipeline {
	agent {dockerfile true}
	stages {
		stage ('Clone Repository'){
		/* Cloning the repository for our workspace*/
			steps {
				checkout scm
			}
		}
		stage ('Build Image') {
			steps {
				sh 'docker build -t cifar_10_classifier .'
			}
		}
		stage ('Run Container') {
			steps {
				echo 'Running container..'
				sh 'docker run -d -p 8008:8080 --name cifar10 cifar_10_classifier'
			}
		}
		stage ('Testing') {
			steps {
				echo 'Testing..'
			}
		}
	}
}