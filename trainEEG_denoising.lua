-----------------------------------------------------
--
--        Train EEG 
--        Using DANN, Cov
--        Version 3 : using Data augmentation 2 
--        14/06/2017
--
----------------------------------------------------

require 'optim'
require 'nn'
--require './rdLayer.lua'
--------------------------------
-- Parameter Handling
--------------------------------
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('EEG-DANN-COV')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-learningRate', 0.001, 'learning rate at t=0')
   cmd:option('-batchSize', 18, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-maxEpoch', 100, 'maximum nb of epoch')
   cmd:option('-seed', 0, 'random seed')
   cmd:option('-domainLambda', 0.1, 'regularization term for transfer learning')
   cmd:option('-saveModel', false, 'flag for saving the model on disk at each epoch, if improvement')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-gpu', 1, 'nombre des Gpus')
   cmd:text()
   opt = cmd:parse(arg or {})
end

torch.manualSeed(opt.seed)

if opt.gpu>0 then
   print('-----------CUDA ON-----------')
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpu)
end
-- 9 sujet 
classes = {'1','2','3','4','5','6','7','8','9'}

-- This matrix records the current confusion across classes
trainConfusion = optim.ConfusionMatrix(classes)
validConfusion = optim.ConfusionMatrix(classes)

----------------------------------------------
--     Function useful
----------------------------------------------
function split(inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={} ; i=1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end

function Generate(trainTensor,labelTrainTensor,validTensor,labelValidTensor,indxDom,Size)
	nSujet = Size
	s=1
	while (s<= nSujet) do -- s number of subject
        	filename=string.format("./data/%s%u%s%u%s","S",s,"_D",indxDom,"_specgramCH_27.csv")
        	file=io.open(filename,"r") -- open file 
        	j=1 -- frequence 
        	t=1 -- time
        	for line in io.lines(filename) do -- line for time  
                	lineSplit=split(line,sep)
       	        	nLineSplit=table.getn(lineSplit)
			
			if t <= nTimeTrain then
                                for frq=1,nLineSplit  do
                                        trainTensor[s][t][frq] = lineSplit[frq]
                                end
                        else
                                for frq=1,nLineSplit  do
                                        ValidTensor[s][t-nTimeTrain][frq] = lineSplit[frq]
                                end
                        t=t+1
                        end
		end
		labelTrainTensor[s] = s
                labelValidTensor[s] = s
		
        	s=s+1
		io.close(file)
	end
end 

function zeroMean(tensor,size,nTime)

	for n=1,size do
        	for t=1,nTime do
                	s=0
                	for frq=1,nFreq do
                        	s=s+tensor[n][t][frq]
                	end
                	for frq=1,nFreq do
                        	tensor[n][t][frq]=tensor[n][t][frq]-(s/nFreq)
                	end
        	end
	end
end

function dataAugmentationSujet (trainTensor,labelTensor,tensor,nbSujet,posIn,nTime)
	pos = posIn
	time=1
	for t=1,nTime do
		for frq=1, nFreq do
			trainTensor[pos][time][frq] = tensor[nbSujet][t][frq]
		end 
		time = time + 1
		if time > ntime then
	       		time =1
			labelTensor[pos]=nbSujet
			pos =pos + 1 
		end 
	end
end

function dataAugmentationSujet2 (resTrainTensor,resLabelTensor,tensor,tensorLabel,posIn,pos,nTime,ntime)
	i = posIn
	j = pos
	bord = posIn+ nTime/ntime -1

	while (i<=bord) do
		if i == posIn + nTime/ntime then
			
			for t=1,ntime-1 do 
                                resTrainTensor[j][t]:copy(tensor[i][t+1])
                        end
			
			resTrainTensor[j][t+1]:copy(tensor[posIn][1])
		else
		
			for t=1, ntime do
				if t == ntime then 
					resTrainTensor[j][t]:copy(tensor[i+1][1])
				else
					resTrainTensor[j][t]:copy(tensor[i][t+1])
				end
			end
			
		end

		resLabelTensor[j]=tensorLabel[posIn]
		i = i + 1
		j = j + 1
		
	end
end		
function alea(tensor, tensorLabel)
	shuffle = torch.randperm(tensor:size(1))	

	aux=torch.CudaTensor(1,tensor:size(2),tensor:size(3))

	for i=1, tensor:size(1) do 
		aux:copy(tensor[shuffle[i]])
		tensor[shuffle[i]]:copy(tensor[i])
		tensor[i]:copy(aux)
		auxLabel=tensorLabel[i]
		tensorLabel[i]=tensorLabel[shuffle[i]]
		tensorLabel[shuffle[i]]=auxLabel	
	end	

end

function noiseAdd(tensor,tensor2,nbSujet,pos,nTime)
        noise=torch.CudaTensor(nTime,nFreq)

        noise:normal(0,0.001)
        for t=1,nTime do
                for frq=1,nFreq do
                        tensor[pos][t][frq]=tensor2[nbSujet][t][frq]+noise[t][frq]
                end
        end

end

sep=','
-------------------------------------------------
--   Data
------------------------------------------------
-- Source Domain
nSourceSujet = 9 
nTimeTrain = 78
nTimeValid = 24
nFreq = 580
ntime=6
ntime2=6
nSourceTrain = 2*(nTimeTrain/ntime)*nSourceSujet
nSourceValid = 2*(nTimeValid/ntime2)*nSourceSujet

sourceTrainSet = torch.CudaTensor(nSourceTrain,ntime,nFreq):zero()
sourceTrainLabel = torch.CudaTensor(nSourceTrain):zero()
sourceValidSet = torch.CudaTensor(nSourceValid,ntime2,nFreq):zero()
sourceValidLabel = torch.CudaTensor(nSourceValid):zero()

print('==> Generate function Source Data')
sourceT = torch.CudaTensor(nSourceTrain,nTimeTrain,nFreq):zero()
sourceTLabels = torch.CudaTensor(nSourceTrain):zero()
sourceV = torch.CudaTensor(nSourceTrain,nTimeValid,nFreq):zero()
sourceVLabels = torch.CudaTensor(nSourceTrain):zero()

Generate(sourceT,sourceTLabels,sourceV,sourceVLabels,2,nSourceSujet)

sourceInputs = torch.CudaTensor(opt.batchSize,sourceTrainSet:size(2),sourceTrainSet:size(3)):zero()
sourceLabels = torch.CudaTensor(opt.batchSize):zero()

-- Target Domain
nTargetSujet = 9

nTargetTrain = 2*(nTimeTrain/ntime)*nTargetSujet
nTargetValid = 2*(nTimeValid/ntime2)*nTargetSujet

targetTrainSet = torch.CudaTensor(nTargetTrain,ntime,nFreq):zero()
targetTrainLabel = torch.CudaTensor(nTargetTrain):zero()
targetValidSet = torch.CudaTensor(nTargetValid,ntime2,nFreq):zero()
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
        posIn=posIn+(nTimeValid/ntime2)
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
        posIn=posIn+(nTimeValid/ntime2)
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
        dataAugmentationSujet2(sourceValidSet,sourceValidLabel,sourceValidSet,sourceValidLabel,posIn,pos,nTimeValid,ntime2)
        posIn = posIn + nTimeValid/ntime2
        pos = pos + nTimeValid/ntime2
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
        dataAugmentationSujet2 (targetValidSet,targetValidLabel,targetValidSet,targetValidLabel,posIn,pos,nTimeValid,ntime2)
        posIn = posIn + nTimeValid/ntime2
        pos = pos + nTimeValid/ntime2
end

---------------------------------------------------
--     Alea
---------------------------------------------------
alea(sourceTrainSet,sourceTrainLabel)
alea(sourceValidSet,sourceValidLabel)
alea(targetTrainSet,targetTrainLabel)
alea(targetValidSet,targetValidLabel)
alea(sourceTrainSet,sourceTrainLabel)
alea(sourceValidSet,sourceValidLabel)
alea(targetTrainSet,targetTrainLabel)
alea(targetValidSet,targetValidLabel)
---------------------------------------------------
--   Definition of the model 
---------------------------------------------------
print('==> model')
-- ModelAv
modelAv = nn.Sequential()
modelAv:add(nn.SpatialConvolution(1,10 ,3, 3, 2, 1, 0, 1))
modelAv:add(nn.ReLU(true))
modelAv:add(nn.SpatialConvolution(10,10 ,3, 3, 1, 1, 0, 1))
modelAv:add(nn.ReLU(true))
modelAv:add(nn.SpatialMaxPooling(2, 2, 2, 1, 1, 0))
modelAv:add(nn.SpatialConvolution(10, 17, 3, 3, 2, 1, 0, 1))
modelAv:add(nn.ReLU(true))
modelAv:add(nn.SpatialConvolution(17, 17, 3, 3, 1, 1, 0, 1))
modelAv:add(nn.ReLU(true))
modelAv:add(nn.SpatialMaxPooling(2, 2, 2, 1, 1, 0))
modelAv:add(nn.Reshape(17*4*35))
modelAv:cuda()

sampleSource=torch.CudaTensor(1,ntime,nFreq)
sampleSource:copy(sourceTrainSet[1])
feats = modelAv:forward(sampleSource)
print(feats:size())

-- ModelPred 
modelPred = nn.Sequential()
modelPred:add(nn.Linear(17*4*35,9))
modelPred:add(nn.Dropout(0.5))
modelPred:add(nn.LogSoftMax())
modelPred:cuda()

--ModelClas
modelClas=nn.Sequential()
modelClas:add(nn.GradientReversal())
modelClas:add(nn.Linear(17*4*35,1))
modelClas:add(nn.Dropout(0.6))
modelClas:add(nn.Sigmoid())
modelClas:cuda()
--------------------------------------
-- Definition of the criterion
--------------------------------------

criterion = nn.ClassNLLCriterion()
criterionClas = nn.BCECriterion()
criterion:cuda()
criterionClas:cuda()

-- Retrieve the pointers to the parameters and gradParameters from the model for lat
featExtractorParams,featExtractorGradParams = modelAv:getParameters()
labelPredictorParams,labelPredictorGradParams = modelPred:getParameters()
domainClassifierParams,domainClassifierGradParams = modelClas:getParameters()
featExtractorParams:cuda()
featExtractorGradParams:cuda()
labelPredictorParams:cuda()
labelPredictorGradParams:cuda()
domainClassifierParams:cuda()
domainClassifierGradParams:cuda()

params = torch.CudaTensor(featExtractorParams:size(1)+labelPredictorParams:size(1)+domainClassifierParams:size(1))
params:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
params:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
params:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)):copy(domainClassifierParams)

gradParams = torch.CudaTensor(featExtractorParams:size(1)+labelPredictorParams:size(1)+domainClassifierParams:size(1))

----------------------------------------
--   Learning function
---------------------------------------

function train()
	print('===> Training')
	tick1 = sys.clock()
	shuffle = torch.randperm(sourceTrainSet:size(1))
	for t = 1,sourceTrainSet:size(1),opt.batchSize do
             xlua.progress(t,sourceTrainSet:size(1))
	     -- Define the minibatch
	     k=1
             for i = 1,opt.batchSize do
		nbSujet = shuffle[k]
                noiseAdd(sourceInputs,sourceTrainSet,nbSujet,i,sourceTrainSet:size(2))
                sourceLabels[i] = sourceTrainLabel[nbSujet]

               	targetInputs[i]:copy(targetTrainSet[shuffle[t+i-1]])
               	targetLabels[i] = targetTrainLabel[shuffle[t+i-1]]
		k=k+1
                if (k>sourceTrainSet:size(1)) then k=1 end
             end
          -- Definition of the evaluation function (closure)
      	  feval = function(x)
	     if params~=x then
                featExtractorParams:copy(x:narrow(1,1,featExtractorParams:size(1)))
                labelPredictorParams:copy(x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)))           
	 	domainClassifierParams:copy(x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)))
	     end
	     
	     featExtractorGradParams:zero()
             labelPredictorGradParams:zero()
             domainClassifierGradParams:zero()
	     for j=1, opt.batchSize do
                 sampleSource=torch.CudaTensor(1,ntime,nFreq)
                 sampleSource:copy(sourceInputs[j])

                 feats = modelAv:forward(sampleSource)
                 preds = modelPred:forward(feats)

                 loss1 = criterion:forward(preds,sourceLabels[j])

                 predsCost  = criterion:backward(preds,sourceLabels[j])

                 labelDfdo = criterion:backward(preds, sourceLabels[j])
                 gradLabelPredictor = modelPred:backward(feats, labelDfdo)
                 modelAv:backward(sampleSource, gradLabelPredictor)

		 domPreds = modelClas:forward(feats)
                 domCost = criterionClas:forward(domPreds,torch.CudaTensor(domPreds:size(1),1):fill(1))
                 domDfdo = criterionClas:backward(domPreds,torch.CudaTensor(domPreds:size(1),1):fill(1))
                 gradDomainClassifier = modelClas:backward(feats,domDfdo,opt.domainLambda)
                 modelAv:backward(sampleSource, gradDomainClassifier,opt.domainLambda)

       		 sampleTarget=torch.CudaTensor(1,ntime,nFreq)
                 sampleTarget:copy(targetInputs[j])

                 targetFeats = modelAv:forward(sampleTarget)
                 targetDomPreds = modelClas:forward(targetFeats)
                 targetDomCost = criterionClas:forward(targetDomPreds,torch.CudaTensor(targetDomPreds:size(1),1):fill(0))
                 targetDomDfdo = criterionClas:backward(targetDomPreds,torch.CudaTensor(targetDomPreds:size(1),1):fill(0))
                 targetGradDomainClassifier = modelClas:backward(targetFeats,targetDomDfdo,opt.domainLambda)
                 modelAv:backward(sampleTarget, targetGradDomainClassifier,opt.domainLambda)

		 params:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
                 params:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
		 params:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)):copy(domainClassifierParams)
		 gradParams:narrow(1,1,featExtractorGradParams:size(1)):copy(featExtractorGradParams)
		 gradParams:narrow(1,featExtractorGradParams:size(1),labelPredictorGradParams:size(1)):copy(labelPredictorGradParams)
		 gradParams:narrow(1,featExtractorGradParams:size(1)+labelPredictorParams:size(1),domainClassifierGradParams:size(1)):copy(domainClassifierGradParams)

	       end	
	     return params,gradParams	
	  end
	  optim.asgd(feval,params,opt)	
	end
	print("tick" .. sys.clock()-tick1)   
end

out = assert(io.open("./ResultsV5.csv", "w"))
out:write("Epoch     trainLoss     trainGlobalValid     validLoss     validGlobalValid")
out:write("\n")
out:write("\n")
prevLoss =10e12
splitter = ","
for i=1, opt.maxEpoch do 
	-- Evaluating the model
	------------------------------
	modelAv:evaluate()
	modelPred:evaluate()
	modelClas:evaluate()

	tensorConfusion=torch.zeros(sourceTrainSet:size(1),9):cuda()

	-------------------------------
	trainLoss=0
	for i=1,sourceTrainSet:size(1) do 
		expSource=torch.CudaTensor(1,ntime,nFreq)
		expSource:copy(sourceTrainSet[i])
		
		sourceFeats = modelAv:forward(expSource)
		sourceTrainPred = modelPred:forward(sourceFeats)
		sourceTrainLoss = criterion:forward(sourceTrainPred, sourceTrainLabel[i])
		sourceDomPreds = modelClas:forward(sourceFeats)
		sourceDomCost = criterionClas:forward(sourceDomPreds,torch.CudaTensor(sourceDomPreds:size(1),1):fill(1))

		expTarget=torch.CudaTensor(1,ntime,nFreq)
		expTarget:copy(targetTrainSet[i])
		targetFeats = modelAv:forward(expTarget)
		targetDomPreds = modelClas:forward(targetFeats)
		targetDomCost = criterionClas:forward(targetDomPreds,torch.CudaTensor(targetDomPreds:size(1),1):fill(0))

		tensorConfusion[i]:copy(sourceTrainPred)
		trainLoss=trainLoss+sourceTrainLoss
	end

	trainConfusion:batchAdd(tensorConfusion, sourceTrainLabel)

	print("EPOCH: " .. i)
	print(trainConfusion)

	tensorConfusion2 = torch.zeros(sourceValidSet:size(1),9):cuda()
	validLoss=0
	for j=1,sourceValidSet:size(1) do
		
		expValid=torch.CudaTensor(1,ntime2,nFreq)
		expValid:copy(sourceValidSet[j])
		
		validPred = modelPred:forward(modelAv:forward(expValid))
		validLoss = criterion:forward(validPred, sourceValidLabel[j])
		sourceFeats = modelAv:forward(expValid)
		sourceValidPred = modelPred:forward(sourceFeats)
		sourceValidLoss = criterion:forward(sourceValidPred, sourceValidLabel[j])
		sourceDomPreds = modelClas:forward(sourceFeats)
		sourceDomCostValid = criterionClas:forward(sourceDomPreds,torch.CudaTensor(sourceDomPreds:size(1),1):fill(1))
		
		expValidTarget=torch.CudaTensor(1,ntime2,nFreq)
		expValidTarget:copy(targetValidSet[i])
		targetFeats = modelAv:forward(expValidTarget)
		targetDomPreds = modelClas:forward(targetFeats)
		targetDomCostValid = criterionClas:forward(targetDomPreds,torch.CudaTensor(targetDomPreds:size(1),1):fill(0))
		tensorConfusion2[j]:copy(validPred)
		validLoss=validLoss + sourceValidLoss

	end
	
	
	validConfusion:batchAdd(tensorConfusion2, sourceValidLabel)
	print(validConfusion)
	print(" + Valid loss " .. validLoss .. " " .. sourceDomCostValid+targetDomCostValid)
	
	out:write(i)
	out:write(splitter)
	out:write(trainLoss)
	out:write(splitter)
	out:write(trainConfusion.totalValid*100)
	out:write(splitter)
	out:write(validLoss)
	out:write(splitter)
	out:write(validConfusion.totalValid*100)
	out:write("\n")
	if i == opt.maxEpoch then
                matrix = assert(io.open("./MatrixConfusionV5.csv", "w"))
                for i=1, trainConfusion.mat:size(1) do
                        for j=1, trainConfusion.mat:size(2) do
                                matrix:write(trainConfusion.mat[i][j])
                                matrix:write(splitter)
                        end
                        matrix:write("\n")
                end
                matrix:close()

                matrix2 = assert(io.open("./MatrixConfusionValidV5.csv", "w"))
                for i=1, trainConfusion.mat:size(1) do
                        for j=1, trainConfusion.mat:size(2) do
                                matrix2:write(trainConfusion.mat[i][j])
                                matrix2:write(splitter)
                        end
                        matrix2:write("\n")
                end
                matrix2:close()
        end

   	--trainLogger:add{i, trainLoss, trainConfusion.totalValid * 100, validLoss, validConfusion.totalValid * 100}
   	trainConfusion:zero()
   	validConfusion:zero()
	
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
	modelAv:training()
	modelPred:training()
	modelClas:training()

	train()

end

out:close()

