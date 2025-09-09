pipeline {
    agent any

    environment {
        AWS_REGION = 'us-east-1'
        ECR_REPO = '047719629738.dkr.ecr.us-east-1.amazonaws.com/my-repo'
        IMAGE_TAG = "build-${BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/priyankas247/RAG-MEDICAL-CHATBOT.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    sh """
                        docker build -t ${ECR_REPO}:${IMAGE_TAG} .
                    """
                }
            }
        }

        stage('Trivy Scan') {
            steps {
                script {
                    sh """
                        docker run --rm \
                          -v /var/run/docker.sock:/var/run/docker.sock \
                          -v $WORKSPACE:/root/.cache/ aquasec/trivy:latest \
                          image --timeout 15m \
                          --severity HIGH,CRITICAL \
                          --format json \
                          -o trivy-report.json \
                          ${ECR_REPO}:${IMAGE_TAG}
                    """
                }
            }
        }

        stage('Push to ECR') {
            steps {
                script {
                    // Login to AWS ECR
                    sh """
                        aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO}
                    """

                    // Retry docker push up to 3 times if it fails
                    retry(3) {
                        timeout(time: 30, unit: 'MINUTES') {
                            sh """
                                docker push ${ECR_REPO}:${IMAGE_TAG}
                            """
                        }
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Archiving Trivy report and cleaning up Docker...'
            archiveArtifacts artifacts: 'trivy-report.json', fingerprint: true
            sh 'docker system prune -f'
        }
    }
}




   //  stage('Deploy to AWS App Runner') {
        //     steps {
        //         withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-token']]) {
        //             script {
        //                 def accountId = sh(script: "aws sts get-caller-identity --query Account --output text", returnStdout: true).trim()
        //                 def ecrUrl = "${accountId}.dkr.ecr.${env.AWS_REGION}.amazonaws.com/${env.ECR_REPO}"
        //                 def imageFullTag = "${ecrUrl}:${IMAGE_TAG}"

        //                 echo "Triggering deployment to AWS App Runner..."

        //                 sh """
        //                 SERVICE_ARN=\$(aws apprunner list-services --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn" --output text --region ${AWS_REGION})
        //                 echo "Found App Runner Service ARN: \$SERVICE_ARN"

        //                 aws apprunner start-deployment --service-arn \$SERVICE_ARN --region ${AWS_REGION}
        //                 """
        //             }
        //         }
        //     }
        // }
    
