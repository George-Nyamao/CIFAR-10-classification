pipeline {
	agent { docker { image 'python:3.7' }}
	stages {
		stage('Build'){
			steps {
				echo 'Building the app ...'
				sh 'python --version'
			}
		}
		stage('Test') {
			steps {
				echo 'Testing the code ..'
			}
		}
		stage('Deploy') {
			steps {
				echo 'Deploying the app ..'
			}
		}
	}
}