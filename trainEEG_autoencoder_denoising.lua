-----------------------------------------------------
--
--        Train EEG 
--        Using DANN, Cov
--        Version 3 : using Data augmentation + DAE 
--        14/06/2017
--
----------------------------------------------------

require 'optim'
require 'nn'
require 'function'
require 'torch'
--------------------------------
-- Parameter Handling
--------------------------------
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('DAE')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-learningRate', 0.1, 'learning rate at t=0')
   cmd:option('-batchSize', 162, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-maxEpoch', 150, 'maximum nb of epoch')
   cmd:option('-seed', 0, 'random seed')
   cmd:option('-domainLambda', 0.1, 'regularization term for transfer learning')
   cmd:option('-saveModel', false, 'flag for saving the model on disk at each epoch, if improvement')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-gpu', 1, 'nombre des Gpus')
   cmd:option('-varNoise',0.001, '--')
   cmd:text()
   opt = cmd:parse(arg or {})
end

torch.manualSeed(opt.seed)

if opt.gpu>0 then
   print('-----------CUDA ON-----------')
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   cutorch.setHeapTracking(true)
end

-- 9 sujet 
classes = {'1','2','3','4','5','6','7','8','9'}

-- This matrix records the current confusion across classes
trainConfusion = optim.ConfusionMatrix(classes)
validConfusion = optim.ConfusionMatrix(classes)

----------------------------------------------
--     Function useful
----------------------------------------------
function max2Tensors(tensorT,tensorV)
	max1=torch.max(tensorT)
	max2=torch.max(tensorV)
	if max1>max2 then 
		max=max1
	else
		max=max2
	end
	return max
end
function min2Tensors(tensorT,tensorV)
	min1=torch.min(tensorT)
	min2=torch.min(tensorV)
	if min1>min2 then 
		min=min1
	else
		min=min2
	end
	return min
end
function normalisation(tensorT,tensorV)
	max = max2Tensors(tensorT,tensorV)
	min = min2Tensors(tensorT,tensorV)

	for k=1,nSourceSujet do 
		max = max2Tensors(tensorT[k],tensorV[k])
		min = min2Tensors(tensorT[k],tensorV[k])

		for i=1, tensorT:size(2) do 
			for j=1,tensorT:size(3) do 
				tensorT[k][i][j] = (tensorT[k][i][j]-min)/(max-min)
			end
		end

		for i=1,tensorV:size(2) do 
			for j=1,tensorV:size(3) do 
				tensorV[k][i][j] = (tensorV[k][i][j]-min)/(max-min)
			end
		end
	end
end

-------------------------------------------------
--   Data
------------------------------------------------
-- Source Domain
nSourceSujet = 9
nTimeTrain = 495
nTimeValid = 85
nFreq = 102
ntime=5
nTime=580
nSourceTrain = 2*(nTimeTrain/ntime)*nSourceSujet
nSourceValid = 2*(nTimeValid/ntime)*nSourceSujet

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

sourceInputs = 
torch.CudaTensor(opt.batchSize,sourceTrainSet:size(2),sourceTrainSet:size(3)):zero()
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

targetInputs = 
torch.CudaTensor(opt.batchSize,targetTrainSet:size(2),targetTrainSet:size(3)):zero()
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
-- Normalisation
---------------------------------------------------
--print('==> Normalisation')
--normalisation(sourceT,sourceV)
---------------------------------------------------
--    Data augmentation
---------------------------------------------------
print('==> Data augmentation')

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

---------------------------------------------------
--    Data Augmentation 2
---------------------------------------------------
print ('==> Data Augmentation 2')
-- sourceTrainSet + sourceTrainLabel
posIn = 1 
pos = nSourceTrain/2 +1
while (posIn <= nSourceTrain/2) do
	dataAugmentationSujet2 (sourceTrainSet,sourceTrainLabel,sourceTrainSet,sourceTrainLabel,posIn,pos,nTimeTrain,ntime)
	posIn = posIn + nTimeTrain/ntime
	pos = pos + nTimeTrain/ntime
end
print('Next 1/3')
-- sourceValidSet +sourceValidLabel
posIn = 1
pos = nSourceValid/2 +1
while (posIn <= nSourceValid/2) do
        dataAugmentationSujet2(sourceValidSet,sourceValidLabel,sourceValidSet,sourceValidLabel,posIn,pos,nTimeValid,ntime)
        posIn = posIn + nTimeValid/ntime
        pos = pos + nTimeValid/ntime
end

print('Next 2/3')

-- targetTrainSet + targetTrainLabel
posIn = 1
pos = nTargetTrain/2 +1
while (posIn <= nTargetTrain/2) do
        dataAugmentationSujet2 (targetTrainSet,targetTrainLabel,targetTrainSet,targetTrainLabel,posIn,pos,nTimeTrain,ntime)
        posIn = posIn + nTimeTrain/ntime
        pos = pos + nTimeTrain/ntime
end
print('Next 3/3')

-- targetValidSet + targetValidLabel
posIn = 1
pos = nTargetValid/2 +1

while (posIn <= nTargetValid/2) do
        dataAugmentationSujet2 (targetValidSet,targetValidLabel,targetValidSet,targetValidLabel,posIn,pos,nTimeValid,ntime)
        posIn = posIn + nTimeValid/ntime
        pos = pos + nTimeValid/ntime
end


---------------------------------------------------
---     Alea
---------------------------------------------------
print('==>Alea')
alea(sourceTrainSet,sourceTrainLabel,targetTrainSet,targetTrainLabel,nSourceTrain)
print('Next')
alea(sourceValidSet,sourceValidLabel,targetValidSet,targetValidLabel,nSourceValid)

---------------------------------------------------
--      Data Mix 
---------------------------------------------------
print('==> data')

dataTrain = torch.CudaTensor(sourceTrainSet:size(1)+targetTrainSet:size(1),ntime,nFreq)
dataLabelTrain = torch.CudaTensor(sourceTrainSet:size(1)+targetTrainSet:size(1))
DataClas(sourceTrainSet,targetTrainSet,dataTrain,dataLabelTrain)
aleaData(dataTrain,dataLabelTrain)

dataValid = torch.CudaTensor(sourceValidSet:size(1)+targetValidSet:size(1),ntime,nFreq)
dataLabelValid = torch.CudaTensor(sourceValidSet:size(1)+targetValidSet:size(1))
DataClas(sourceValidSet,targetValidSet,dataValid,dataLabelValid)
aleaData(dataValid,dataLabelValid)

---------------------------------------------------
--   Definition of the model 
---------------------------------------------------
print('==> model')
--------------1------------------------------------
-- Create encoder
encoder = nn.Sequential()
noiser=nn.WhiteNoise(0,opt.sd)
encoder:add(noiser)
encoder:add(nn.SpatialConvolution(1,50 ,5, 3, 2, 2, 3, 3))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialConvolution(50,50 ,5, 3, 2, 2, 3, 3))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialAveragePooling(2,2,2,2))
encoder:cuda()

decoder = nn.Sequential()
decoder:add(nn.Reshape(50*2*13))
decoder:add(nn.ReLU(true))
decoder:add(nn.Linear(50*2*13,510))
decoder:add(nn.ReLU(true))
decoder:add(nn.Reshape(1,5,102))

--decoder = nn.Sequential()
--decoder:add(nn.SpatialUpSamplingNearest(2))
--decoder:add(nn.ReLU(true))
--decoder:add(nn.Tanh(true))
--decoder:add(nn.SpatialConvolution(50,50 ,5, 3, 1, 2, 15, 2))
--decoder:add(nn.ReLU(true))
--decoder:add(nn.SpatialConvolution(50,ntime,3, 3, 1, 2, 26, 0))
--decoder:add(nn.Tanh(true))
decoder:cuda()

-----------------------------
criterion = nn.SmoothL1Criterion()
criterion:cuda()

-- Retrieve the pointers to the parameters and gradParameters from the model for lat$
ParamsEn,GradParamsEn = encoder:getParameters()
ParamsDe,GradParamsDe = encoder:getParameters()

ParamsDe:cuda()
GradParamsDe:cuda()
ParamsEn:cuda()
GradParamsEn:cuda()

params = torch.CudaTensor(ParamsEn:size(1)+ParamsDe:size(1))
params:narrow(1,1,ParamsEn:size(1)):copy(ParamsEn)
params:narrow(1,ParamsEn:size(1),ParamsDe:size(1)):copy(ParamsDe)

gradParams = torch.CudaTensor(ParamsEn:size(1)+ParamsDe:size(1))

----------------------------------------
--   Learning function
---------------------------------------
dataInputs = torch.CudaTensor(opt.batchSize,1,ntime,nFreq)
learning_rate = 100
optimParams = {learning_rate}

function train()
	print('===> Training')
	tick1 = sys.clock()
	shuffle = torch.randperm(sourceTrainSet:size(1))
	for t = 1,sourceTrainSet:size(1),opt.batchSize do

             xlua.progress(t,sourceTrainSet:size(1))
	     -- Define the minibatch
                for i=1, opt.batchSize do

                        dataInputs[i][1]:copy(dataTrain[shuffle[t+i-1]])
                end
for mm=1,15 do 
	for j=1,opt.batchSize do 
          -- Definition of the evaluation function (closure)
      	  feval = function(x)

	     	ParamsEn:copy(x:narrow(1,1,ParamsEn:size(1)))
             	ParamsDe:copy(x:narrow(1,ParamsEn:size(1),ParamsDe:size(1)))

		 GradParamsEn:zero()
	     	 GradParamsDe:zero()


		 feat = encoder:forward(dataInputs[j])
                 preds = decoder:forward(feat)

		 loss = criterion:forward(preds,dataInputs[j])
		
		 --print(j,'-',loss)
                 predsCost  = criterion:backward(preds,dataInputs[j])

	  	 gradLabelPredictor = decoder:backward(feat,predsCost)

  		 encoder:backward(dataInputs[j],gradLabelPredictor)

		 params:narrow(1,1,ParamsEn:size(1)):copy(ParamsEn)
                 params:narrow(1,ParamsEn:size(1),ParamsDe:size(1)):copy(ParamsDe)

		 gradParams:narrow(1,1,GradParamsEn:size(1)):copy(GradParamsEn)
		 gradParams:narrow(1,GradParamsEn:size(1),GradParamsDe:size(1)):copy(GradParamsDe)
		 --print('---1---',torch.min(GradParamsEn),torch.max(GradParamsEn))
		 --print('---2---',torch.min(GradParamsDe),torch.max(GradParamsDe))

		 --print('---3---',torch.min(gradParams),torch.max(gradParams)) 
	     return params,gradParams	
	  end

	 optim.adam(feval,params,optimParams)
       end	
end
	end

	print("tick" .. sys.clock()-tick1)   
end

out = assert(io.open("./Results_DAE.csv", "w"))
out:write("Epoch     trainLoss     validLoss")
out:write("\n")
out:write("\n")
splitter = ","
dataInputTrain = torch.CudaTensor(dataTrain:size(1),1,ntime,nFreq)
dataInputValid = torch.CudaTensor(dataValid:size(1),1,ntime,nFreq)

for i=1,dataTrain:size(1) do 
	dataInputTrain[i][1]:copy(dataTrain[i])
end

for i=1,dataValid:size(1) do 
	dataInputValid[i][1]:copy(dataValid[i])
end
for i=1, opt.maxEpoch do 

	------------------------------
	-- Evaluating the model
	------------------------------
	encoder:evaluate()
	decoder:evaluate()
	-------------------------------

	trainLoss=0
	for d=1,dataTrain:size(1) do 
		feat = encoder:forward(dataInputTrain[d])
                preds = decoder:forward(feat)
		loss = criterion:forward(preds,dataInputTrain[d])
		trainLoss=trainLoss+loss
	end

	trainLoss=trainLoss/dataInputTrain:size(1)

	print("EPOCH: " .. i)
	print("trainLoss: ",trainLoss)

	validLoss=0
        for d=1,dataValid:size(1) do
		feat = encoder:forward(dataInputValid[d])
                preds = decoder:forward(feat)
		loss = criterion:forward(preds,dataInputValid[d])
		validLoss = validLoss + loss
	end
	validLoss = validLoss/dataInputValid:size(1)
	print("validLoss: ",validLoss)
	out:write(i)
	out:write(splitter)
	out:write(trainLoss)
	out:write(splitter)
	out:write(validLoss)
	out:write("\n")

------------------------------
-- Save model
------------------------------
	if opt.saveModel then
        	if trainLoss < prevLoss then
                	prevLoss = trainLoss
                	torch.save("model.bin",model)
        	else
                	model = torch.load("model.bin")
         	end
	end
	encoder:training()
	decoder:training()

	train()

end
out:close()

