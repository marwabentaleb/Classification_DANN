require 'torch'
require 'nn'
require 'VanillaRNN'
require 'optim'
require 'function'

if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('EEG-DANN-COV')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-learningRate', 0.4, 'learning rate at t=0') 
   cmd:option('-batchSize', 9, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-maxEpoch', 150, 'maximum nb of epoch')
   cmd:option('-seed', 0, 'random seed')
   cmd:option('-saveModel', false, 'flag for saving the model on disk at each epoch, if improvement')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-gpu', 1, 'nombre des Gpus')
   cmd:text()
   opt = cmd:parse(arg or {})
end

torch.manualSeed(opt.seed)
gpu=1
if gpu>0 then
   print('-----------CUDA ON-----------')
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(gpu)
end

-------------------------------------------------
--   Data
------------------------------------------------
-- Source Domain
nSourceSujet = 9 
nTimeTrain = 460
nTimeValid = 120
nFreq = 102
ntime=20
nTime=580
nSourceTrain = (nTimeTrain/ntime)*nSourceSujet
nSourceValid = (nTimeValid/ntime)*nSourceSujet

sourceTrainSet = torch.CudaTensor(nSourceTrain,ntime,nFreq):zero()
sourceTrainLabel = torch.CudaTensor(nSourceTrain):zero()
sourceValidSet = torch.CudaTensor(nSourceValid,ntime,nFreq):zero()
sourceValidLabel = torch.CudaTensor(nSourceValid):zero()
print('==> Generate function Source Data')
sourceT = torch.CudaTensor(nSourceSujet,nTimeTrain,nFreq):zero()
sourceTLabels = torch.CudaTensor(nSourceSujet):zero()
sourceV = torch.CudaTensor(nSourceSujet,nTimeValid,nFreq):zero()
sourceVLabels = torch.CudaTensor(nSourceSujet):zero()

Generate(sourceT,sourceTLabels,sourceV,sourceVLabels,2,nSourceSujet)

sourceInputs = torch.CudaTensor(opt.batchSize,sourceTrainSet:size(2),sourceTrainSet:size(3)):zero()
sourceLabels = torch.CudaTensor(opt.batchSize):zero()

-- Target Domain
nTargetSujet = 9

nTargetTrain = 2*(nTimeTrain/ntime)*nTargetSujet
nTargetValid = 2*(nTimeValid/ntime)*nTargetSujet

targetTrainSet = torch.CudaTensor(nTargetTrain,ntime,nFreq):zero()
targetTrainLabel = torch.CudaTensor(nTargetTrain):zero()
targetValidSet = torch.CudaTensor(nTargetValid,ntime,nFreq):zero()
targetValidLabel = torch.CudaTensor(nTargetValid):zero()

print('==> Generate function Target Data')
targetT = torch.CudaTensor(nTargetSujet,nTimeTrain,nFreq):zero()
targetTLabels = torch.CudaTensor(nTargetSujet):zero()
targetV = torch.CudaTensor(nTargetSujet,nTimeValid,nFreq):zero()
targetVLabels= torch.CudaTensor(nTargetSujet):zero()

Generate(targetT,targetTLabels,targetV,targetVLabels,4,nTargetSujet)

targetInputs = torch.CudaTensor(opt.batchSize,targetTrainSet:size(2),targetTrainSet:size(3)):zero()
targetLabels = torch.CudaTensor(opt.batchSize):zero()


---------------------------------------------------
--    Zero Mean
---------------------------------------------------
print('==> zeroMean function')

-- sourceTrainSet
zeroMean(sourceT,nSourceSujet,sourceT:size(2))
print('Next 1/3')
-- targetTrainSet
zeroMean(targetT,nTargetSujet,targetT:size(2))
print('Next 2/3')
-- sourceValid
zeroMean(sourceV,nSourceSujet,sourceV:size(2))
print('Next 3/3')
--targetValid
zeroMean(targetV,nTargetSujet,targetV:size(2))

---------------------------------------------------
--    Data augmentation
---------------------------------------------------
print('==> Redimensionnement')

-- sourceTrainSet + sourceTrainLabel 
posIn=1
for i=1,nSourceSujet  do 
	dataAugmentationSujet (sourceTrainSet,sourceTrainLabel,sourceT,i,posIn,nTimeTrain)
	posIn=posIn+(nTimeTrain/ntime)
end
print('Next 1/3')

-- sourceValidSet +sourceValidLabel
posIn=1
for i=1,nSourceSujet  do
        dataAugmentationSujet (sourceValidSet,sourceValidLabel,sourceV,i,posIn,nTimeValid)
        posIn=posIn+(nTimeValid/ntime)
end
print('Next 2/3')

-- targetTrainSet + targetTrainLabel
posIn=1
for i=1,nTargetSujet  do
        dataAugmentationSujet (targetTrainSet,targetTrainLabel,targetT,i,posIn,nTimeTrain)
        posIn=posIn+(nTimeTrain/ntime)
end
print('Next 3/3')

-- targetValidSet + targetValidLabel
posIn=1
for i=1,nTargetSujet  do
        dataAugmentationSujet (targetValidSet,targetValidLabel,targetV,i,posIn,nTimeValid)
        posIn=posIn+(nTimeValid/ntime)
end

---------------------------------------------
--maxTensor=torch.CudaTensor(sourceTrainSet:size(1),sourceTrainSet:size(2),1)

--maxObs(sourceTrainSet,maxTensor)
--rescal(sourceTrainSet,maxTensor)

---------------------------------------------
moyenneTensor=torch.CudaTensor(9,sourceTrainSet:size(3))
stdTensor=torch.CudaTensor(9,sourceTrainSet:size(3))
Rescal(sourceTrainSet,sourceValidSet,moyenneTensor,stdTensor)

---------------------------------------------
D=sourceTrainSet:size(3)
H= D/2
rnn = nn.VanillaRNN(D, H)
rnn:cuda()

output = nn.Sequential()
output:add(nn.Linear(H,3/2*D))
output:add(nn.ReLU(true))
output:add(nn.Dropout(0.7))
output:add(nn.Linear(3*D/2,D))
output:cuda()

paramsRNN, grad_paramsRNN = rnn:getParameters()
paramsRNN:cuda()
grad_paramsRNN:cuda()
paramsOut,grad_paramsOut = output:getParameters()
paramsOut:cuda()
grad_paramsOut:cuda()
params = torch.CudaTensor(paramsRNN:size(1) + paramsOut:size(1))
params:narrow(1,1,paramsRNN:size(1)):copy(paramsRNN)
params:narrow(1,paramsRNN:size(1),paramsOut:size(1)):copy(paramsOut)
grad_params = torch.CudaTensor(paramsRNN:size(1) + paramsOut:size(1))

--crit = nn.AbsCriterion()
crit = nn.MSECriterion()
crit:cuda()
N, T = nTimeTrain/ntime-1, sourceTrainSet:size(2)

sourceRNN=torch.CudaTensor(N,T,D)
refRNN=torch.CudaTensor(N,1,D)
nSujet=9
grad_out = torch.CudaTensor(N,H)

function train()
	print('==>Trainning')
	local tick1 = sys.clock()
	for r =1,30 do
	for n=1,nSujet do 
		
		for i = 1,N do
			 sourceRNN[i]:copy(sourceTrainSet[i+(n-1)*N])
		end	 
		function f(w)

			paramsRNN:copy(w:narrow(1,1,paramsRNN:size(1)))
			paramsOut:copy(w:narrow(1,paramsRNN:size(1),paramsOut:size(1)))
			
			grad_paramsRNN:zero()
			grad_paramsOut:zero()

			scores = rnn:forward(sourceRNN)
			
			for l=1,N do
					t=T 
					yPred=output:forward(scores[l][t])
					if t<T then
						loss = crit:forward(yPred,sourceRNN[l][t+1])
	        				grad_scores =crit:backward(yPred,sourceRNN[l][t+1])

					else 
						if l<N then
							loss = crit:forward(yPred,sourceRNN[l+1][1])
	        					grad_scores=crit:backward(yPred,sourceRNN[l+1][1])
						else 
							aux=torch.CudaTensor(nFreq)
							aux:copy(sourceTrainSet[N*n+1][1])
							loss=crit:forward(yPred,aux)
							grad_scores=crit:backward(yPred,aux)
						end
					end
					
					grad_out[l]:copy(output:backward(scores[l][t],grad_scores))
	
		end
			rnn:backward(sourceRNN, grad_out)
			params:narrow(1,1,paramsRNN:size(1)):copy(paramsRNN)
			params:narrow(1,paramsRNN:size(1),paramsOut:size(1)):copy(paramsOut)
			grad_params:narrow(1,1,paramsRNN:size(1)):copy(grad_paramsRNN)
			grad_params:narrow(1,paramsRNN:size(1),paramsOut:size(1)):copy(grad_paramsOut)
			return params, grad_params
		end
	
		optim.adam(f,params,opt)
	end
	end
	print("tick" .. sys.clock()-tick1)
end
out = assert(io.open("./ResultatRnn", "w"))
out:write("Epoch     Loss")
out:write("\n")
out:write("\n")
splitter = ","

N1=nTimeValid/ntime-1
sourceRNN1=torch.CudaTensor(N1,T,D)
refRNN1=torch.CudaTensor(N1,1,D)

for i=1, opt.maxEpoch do 
	-- Evaluating the model 

	rnn:evaluate()
	output:evaluate()

	trainLoss =0
	 for t=1,nSujet do

                for i = 1,N do
                         sourceRNN[i]:copy(sourceTrainSet[i+(t-1)*N])
		end


		for i=1,N-1 do
                        refRNN[i]:copy(sourceRNN[i+1][1])
                end
		aux=torch.CudaTensor(nFreq)
		aux:copy(sourceTrainSet[t*N+1][1])

		refRNN[N]:copy(aux)
		scores = rnn:forward(sourceRNN)

		for l=1,N do
		        yPred=output:forward(scores[l][T])
                	loss =crit:forward(yPred,refRNN[l][1])
        		trainLoss = trainLoss + loss
		end
	end

	trainLoss =trainLoss/(nSujet*N)

	validLoss=0
	for t=1,nSujet do

                for i = 1,N1 do
                         sourceRNN1[i]:copy(sourceValidSet[i+(t-1)*N1])
                end


                for i=1,N1-1 do
                        refRNN1[i]:copy(sourceRNN1[i+1][1])
                end

                aux=torch.CudaTensor(nFreq)
                aux:copy(sourceValidSet[(t*N1)+1][1])

                refRNN[N1]:copy(aux)
                scores = rnn:forward(sourceRNN1)
                for l=1,N1-1 do
			print(yPred)
			print(refRNN1[l][1])
                        yPred=output:forward(scores[l][T])
                        loss =crit:forward(yPred,refRNN1[l][1])
			validLoss = validLoss + loss
                end
        end
	
	validLoss=validLoss/(nSujet*(N1-1))
	out:write(i)
	out:write(splitter)
	out:write(trainLoss)
	out:write(splitter)
	out:write(validLoss)
	out:write("\n")
	
	print('Epoch : ', i, '  trainLoss : ', trainLoss)
	print('Epoch : ', i, '  validLoss : ', validLoss)

	rnn:training()
	train()

end

out:close()
