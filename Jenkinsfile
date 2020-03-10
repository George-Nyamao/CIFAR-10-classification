pipeline {
    agent { dockerfile true }
	stages {
		stage ('Clone Repository'){
			steps {
				checkout scm
			}
		}
		stage ('Build Image') {
			steps {
				sh 'docker build --tag=cifar_10_jenkins .'
			}
		}
		stage ('Run Container') {
			steps {
				echo 'Running container..'
				sh 'docker run -d -p 8008:8080 cifar_10_jenkins'
			}
		}
		stage ('Open Browser') {
			steps {
				echo 'opening..'
                sh 'google-chrome http://127.0.0.1:8008'
			}
		}
	}
}