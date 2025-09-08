pipeline {
    agent any

    environment {
        AWS_REGION = 'us-east-1'
        ECR_REPO = '047719629738.dkr.ecr.us-east-1.amazonaws.com/my-repo'
        IMAGE_TAG = "build-${BUILD_NUMBER}"
        TRIVY_CACHE = '/tmp/trivy-cache'
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/priyankas247/RAG-MEDICAL-CHATBOT.git'
            }
        }

        stage('Login to AWS ECR') {
            steps {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-token']]) {
                    sh """
                        aws ecr get-login-password --region ${AWS_REGION} \
                          | docker login --username AWS --password-stdin ${ECR_REPO}
                    """
                }
            }
        }

        stage('Docker Build') {
            steps {
                sh """
                    docker pull ${ECR_REPO}:latest || true
                    docker build \
                        --cache-from=${ECR_REPO}:latest \
                        -t ${ECR_REPO}:${IMAGE_TAG} \
                        -t ${ECR_REPO}:latest .
                """
            }
        }

        stage('Trivy Scan') {
            steps {
                sh """
                    mkdir -p ${TRIVY_CACHE}
                    trivy image --cache-dir ${TRIVY_CACHE} --light \
                        --timeout 5m --severity HIGH,CRITICAL \
                        --format json -o trivy-report.json ${ECR_REPO}:${IMAGE_TAG} || true
                """
            }
        }

        stage('Docker Push') {
            steps {
                sh """
                    docker push ${ECR_REPO}:${IMAGE_TAG}
                    docker push ${ECR_REPO}:latest
                """
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'trivy-report.json', allowEmptyArchive: true
            echo 'Cleaning up Docker...'
            sh 'docker system prune -af --volumes || true'
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
    
