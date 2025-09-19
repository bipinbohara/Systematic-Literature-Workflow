# PaperSummarizer

###
Login to your account on the GPU cluster, then:
```
Clone this repository: https://github.com/bipinbohara/Systematic-Literature-Workflow.git
```

## Step 1:
update the deployment.yaml file with your name in place of <your-name-here> and then create a deployment as follows:
```
kubectl apply -f deployment.yaml
```

## Step 2:
```
kubectl exec -it paper-summary -n workspace-<your-name-here> -- /bin/bash
```

## Step 3: Go to the following directory to find the file run_execution.sh
```
cd mnt/data/PaperSummary
```

## Step 4: run the following script
```
chmode +x run_execution.sh

./run_execution.sh
```

<br>

### Copy files/directory:
From local machine to GPU cluster account: Execute this from your local machine
```
scp -r "/Users/username/Desktop/ResearchPapers/" sys-head-admin@131.151.54.248:~"/Desktop/ResearchPapers/"
```

From GPU Machine to container: execute this from the GPU cluster/ Head Node
```
kubectl cp "/home/<your-user-in-GPU-Cluster>/Desktop/Papers/." workspace-<your-name-here>/paper-summary:/mnt/data/PaperSummarizer/data -c python
```

<br>

### Copy output file back to your local machine: Execute this from the GPU cluster/ Head Node
```
kubectl cp workspace-<your-name-here>/paper-summary:/mnt/data/PaperSummarizer/output/. -c python "/home/<your-user-in-GPU-Cluster>/Desktop/Papers/Output/."
```
Local terminal:
```
scp -r <your-user-name-GPU-Cluster>@131.151.54.248:~"/Desktop/Papers/Output/." ~"/Desktop/Research Papers/Output"
```
