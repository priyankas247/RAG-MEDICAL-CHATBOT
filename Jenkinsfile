pipeline {
    agent any

    environment {
        AWS_REGION = 'us-east-1'                  // 🛠️ Change as needed
        ECR_REPO = 'my-repo'                      // 🛠️ Your ECR repo name
        IMAGE_TAG = 'latest'
    }

    stages {
        stage('Build, Scan, and Push Docker Image to ECR') {
            steps {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-token']]) {
                    script {
                        def accountId = sh(
                            script: "aws sts get-caller-identity --query Account --output text",
                            returnStdout: true
                        ).trim()

                        def ecrUrl = "${accountId}.dkr.ecr.${env.AWS_REGION}.amazonaws.com/${env.ECR_REPO}"
                        def imageFullTag = "${ecrUrl}:${IMAGE_TAG}"

                        sh """
                        echo "🔐 Logging into AWS ECR..."
                        aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ecrUrl}

                        echo "🐳 Building Docker image..."
                        docker build -t ${env.ECR_REPO}:${IMAGE_TAG} .

                        echo "🔍 Scanning Docker image with Trivy..."
                        docker run --rm \\
                            -v /var/run/docker.sock:/var/run/docker.sock \\
                            -v \"${env.WORKSPACE}:/root\" \\
                            aquasec/trivy \\
                            image --scanners vuln \\
                            --severity HIGH,CRITICAL \\
                            --timeout 10m \\
                            --format json \\
                            -o /root/trivy-report.json \\
                            ${env.ECR_REPO}:${IMAGE_TAG} || true

                        echo "📦 Tagging and pushing Docker image to ECR..."
                        docker tag ${env.ECR_REPO}:${IMAGE_TAG} ${imageFullTag}
                        docker push ${imageFullTag}
                        """
                    }
                }
            }
        }

        stage('Archive Trivy Report') {
            steps {
                script {
                    // Debug to confirm file exists
                    sh 'ls -lh "${WORKSPACE}" || true'
                    sh 'cat "${WORKSPACE}/trivy-report.json" || echo "⚠️ Report not found"'
                }

                // Archive the file
                archiveArtifacts artifacts: 'trivy-report.json', allowEmptyArchive: true
            }
        }

        // Optional Deployment Stage
        // stage('Deploy to AWS App Runner') {
        //     steps {
        //         withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-token']]) {
        //             script {
        //                 def accountId = sh(script: "aws sts get-caller-identity --query Account --output text", returnStdout: true).trim()
        //                 def ecrUrl = "${accountId}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
        //                 def imageFullTag = "${ecrUrl}:${IMAGE_TAG}"
        //
        //                 echo "Triggering deployment to AWS App Runner..."
        //
        //                 sh """
        //                     SERVICE_ARN=\$(aws apprunner list-services --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn" --output text --region ${AWS_REGION})
        //                     echo "Found App Runner Service ARN: \$SERVICE_ARN"
        //                     aws apprunner start-deployment --service-arn \$SERVICE_ARN --region ${AWS_REGION}
        //                 """
        //             }
        //         }
        //     }
        // }
    }
}
