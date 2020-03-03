pipeline {
	agent { docker { image 'python:3.7' }}
	stages {
		stage('Build'){
			steps {
				echo 'Building the app ...'
				sh 'env\Scripts\activate'
				sh 'pip install -r requiremets.txt"	
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
	post {
        always {
            echo 'This will always run'
        }
        success {
            echo 'This will run only if successful'
        }
        failure {
            echo 'This will run only if failed'
        }
        unstable {
            echo 'This will run only if the run was marked as unstable'
        }
        changed {
            echo 'This will run only if the state of the Pipeline has changed'
            echo 'For example, if the Pipeline was previously failing but is now successful'
        }
    }
}