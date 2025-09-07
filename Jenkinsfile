pipeline { 
    agent any

    environment {
        AWS_REGION   = 'us-east-1'
        ECR_REPO     = 'my-repo'
        IMAGE_TAG    = 'latest'
        SERVICE_NAME = 'llmops-medical-service'
    }

    stages {
        stage('Clone GitHub Repo') {
            steps {
                echo '📥 Cloning GitHub repo to Jenkins...'
                checkout scmGit(
                    branches: [[name: '*/main']],
                    extensions: [],
                    userRemoteConfigs: [[
                        credentialsId: 'github-token',
                        url: 'https://github.com/priyankas247/RAG-MEDICAL-CHATBOT.git'
                    ]]
                )
            }
        }

        stage('Build, Scan, and Push Docker Image to ECR') {
            steps {
                withAWS(credentials: 'aws-token', region: "${AWS_REGION}") {
                    script {
                        def accountId    = sh(script: "aws sts get-caller-identity --query Account --output text", returnStdout: true).trim()
                        def ecrUrl       = "${accountId}.dkr.ecr.${env.AWS_REGION}.amazonaws.com/${env.ECR_REPO}"
                        def imageFullTag = "${ecrUrl}:${IMAGE_TAG}"

                        sh """
                        echo "🔐 Logging into AWS ECR..."
                        aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ecrUrl}

                        echo "🐳 Building Docker image..."
                        docker build -t ${env.ECR_REPO}:${IMAGE_TAG} .

                        echo "🔍 Scanning Docker image with Trivy..."
                        docker run --rm \
                            -v /var/run/docker.sock:/var/run/docker.sock \
                            -v ${env.WORKSPACE}:${env.WORKSPACE} \
                            -w ${env.WORKSPACE} \
                            aquasec/trivy image \
                            --timeout 30m \
                            --skip-db-update=false \
                            --scanners vuln \
                            --severity HIGH,CRITICAL \
                            --format json \
                            -o ${env.WORKSPACE}/trivy-report.json \
                            ${env.ECR_REPO}:${IMAGE_TAG} || echo '{}' > ${env.WORKSPACE}/trivy-report.json

                        echo "📦 Tagging and pushing Docker image..."
                        docker tag ${env.ECR_REPO}:${IMAGE_TAG} ${imageFullTag}
                        docker push ${imageFullTag}
                        """
                    }

                    // Archive security scan report
                    archiveArtifacts artifacts: 'trivy-report.json', allowEmptyArchive: true
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
