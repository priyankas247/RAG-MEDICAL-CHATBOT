pipeline {
    agent any

    environment {
        AWS_REGION = 'us-east-1'
        ECR_REPO = '047719629738.dkr.ecr.us-east-1.amazonaws.com/rag-medical-chatbot'
        IMAGE_TAG = "build-${BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/priyankas247/RAG-MEDICAL-CHATBOT.git'
            }
        }

        stage('Build, Scan, and Push Docker Image to ECR') {
            steps {
                script {
                    withAWS(region: "${AWS_REGION}", credentials: 'aws-creds-id') {
                        sh '''
                            echo "🔹 Logging in to Amazon ECR..."
                            aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO

                            echo "🔹 Building Docker image..."
                            docker build -t $ECR_REPO:$IMAGE_TAG .

                            echo "🔹 Pushing Docker image to ECR..."
                            docker push $ECR_REPO:$IMAGE_TAG
                        '''
                    }
                }
            }
        }
    }
}


        // stage('Deploy to AWS App Runner') {
        //     steps {
        //         withAWS(credentials: 'aws-token', region: "${AWS_REGION}") {
        //             script {
        //                 def accountId    = sh(script: "aws sts get-caller-identity --query Account --output text", returnStdout: true).trim()
        //                 def ecrUrl       = "${accountId}.dkr.ecr.${env.AWS_REGION}.amazonaws.com/${env.ECR_REPO}"
        //                 def imageFullTag = "${ecrUrl}:${IMAGE_TAG}"

        //                 echo "🚀 Triggering deployment to AWS App Runner..."

        //                 sh """
        //                 SERVICE_ARN=\$(aws apprunner list-services --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn" --output text --region ${AWS_REGION})
        //                 echo "Found App Runner Service ARN: \$SERVICE_ARN"

        //                 aws apprunner start-deployment --service-arn \$SERVICE_ARN --region ${AWS_REGION}
        //                 """
        //             }
        //         }
        //     }
        // }
    }
}
