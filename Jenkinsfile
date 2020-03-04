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
				sh 'sudo docker build -t cifar_10_classifierv1 .'
			}
		}
		stage ('Run Image') {
			steps {
				sh 'sudo docker run -d --name cifar10classifier cifar_10_classifierv1'
			}
		}
		stage ('Testing') {
			steps {
				echo 'Testing..'
			}
		}
	}
}