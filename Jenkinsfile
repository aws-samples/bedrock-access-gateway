pipeline {
  agent {
    kubernetes {
      inheritFrom 'inbound-agent'
      defaultContainer 'jnlp'
      label 'inbound-agent'
    }
  }

  environment {
    // Build version will be generated in the pipeline
    BUILD_VERSION = ""
  }

  stages {
    stage('Build & Push Docker Image') {
      steps {
        script {
          // Generate dynamic build version based on current date
          def currentDate = new Date()
          def year = currentDate.format('yy')  // Last 2 digits of year
          def month = currentDate.format('MM') as Integer
          
          // Determine quarter based on month
          def quarter
          if (month >= 1 && month <= 3) {
            quarter = 1
          } else if (month >= 4 && month <= 6) {
            quarter = 2
          } else if (month >= 7 && month <= 9) {
            quarter = 3
          } else {
            quarter = 4
          }
          
          // Create BUILD_VERSION as local variable first
          def BUILD_VERSION = "${year}.${quarter}-SNAPSHOT-${BUILD_NUMBER}"
          
          // Set it as environment variable for shell scripts
          env.BUILD_VERSION = BUILD_VERSION
          
          echo "Generated BUILD_VERSION: ${BUILD_VERSION}"
          echo "Year: 20${year}, Quarter: ${quarter}, Build Number: ${BUILD_NUMBER}"
          
          sh """#!/bin/bash
            # Install required tools if not available
            if ! command -v docker &> /dev/null; then
              apt-get update
              apt-get install -y docker.io
            fi
            
            if ! command -v aws &> /dev/null; then
              apt-get update
              apt-get install -y awscli jq
            fi
            
            if ! command -v jq &> /dev/null; then
              apt-get install -y jq
            fi
            
            # Build the Docker image
            echo "Building Docker image: bedrock-access-gateway:${BUILD_VERSION}"
            docker build -t bedrock-access-gateway ./src
            
            # Tag image for ECR
            docker tag bedrock-access-gateway:latest 382254873799.dkr.ecr.us-east-1.amazonaws.com/bedrock-access-gateway:${BUILD_VERSION}
            
            # Set AWS region
            export AWS_DEFAULT_REGION=us-east-1
            
            # Assume the staging role for ECR access
            echo "Assuming AWS role for ECR push..."
            TEMP_ROLE=\$(aws sts assume-role \\
              --role-arn "arn:aws:iam::382254873799:role/jenkins_ai-foundation_runway" \\
              --role-session-name "jenkins-docker-push-${BUILD_NUMBER}")
            
            export AWS_ACCESS_KEY_ID=\$(echo \$TEMP_ROLE | jq -r .Credentials.AccessKeyId)
            export AWS_SECRET_ACCESS_KEY=\$(echo \$TEMP_ROLE | jq -r .Credentials.SecretAccessKey)
            export AWS_SESSION_TOKEN=\$(echo \$TEMP_ROLE | jq -r .Credentials.SessionToken)
            
            # Login to ECR
            echo "Logging into ECR..."
            aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 382254873799.dkr.ecr.us-east-1.amazonaws.com
            
            # Push to ECR
            echo "Pushing image to ECR..."
            docker push 382254873799.dkr.ecr.us-east-1.amazonaws.com/bedrock-access-gateway:${BUILD_VERSION}
            
            echo "Successfully pushed image:"
            echo "382254873799.dkr.ecr.us-east-1.amazonaws.com/bedrock-access-gateway:${BUILD_VERSION}"
          """
        }
      }
    }
  }
}