pipeline {
	agent any
	stages {
		stage ('Clone Repository'){
		/* Cloning the repository for our workspace*/
			steps {
				checkout scm
			}
		}
		stage ('Build Image') {
			steps {
				sh 'docker build -t cifar_10_classifier:v1 .'
			}
		}
		stage ('Run Image') {
			steps {
				echo 'Will run soon..'
			}
		}
		stage ('Testing') {
			steps {
				echo 'Testing..'
			}
		}
	}
}