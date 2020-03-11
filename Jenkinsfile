pipeline {
    agent { dockerfile true }
	stages {
		stage ('First'){
			steps {
				checkout scm
			}
		}
		stage ('Build Image') {
			steps {
				echo 'Build stage .. passed!'
			}
		}
		stage ('Run Container') {
			steps {
				echo 'Running container..'
			}
		}
		stage ('Open Browser') {
			steps {
				echo 'opening..'
                sh 'google-chrome http://www.facebook.com'
			}
		}
	}
}