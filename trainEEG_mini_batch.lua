-----------------------------------------------------
--
--        Train EEG 
--        Using DANN, Cov
--        Version 3 : using Data augmentation 2 
--        14/06/2017
--
----------------------------------------------------
require 'torch'
require 'optim'
require 'nn'
require 'function'
-----------------------------------------------------------------------------

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
   cmd:option('-learningRate', 0.00000001, 'learning rate at t=0') 
   cmd:option('-rateRNN',0.4,'learning rate for RNN at t=0')
   cmd:option('-batchSize', 9, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-maxEpoch', 150, 'maximum nb of epoch for DANN')
   cmd:option('-seed', 0, 'random seed')
   cmd:option('-domainLambda', 0.1, 'regularization term for transfer learning')
   cmd:option('-saveModel', false, 'flag for saving the model on disk at each epoch, if improvement')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-gpu', 1, 'nombre des Gpus')
   cmd:option('-sd',0.01,'noise')
   cmd:option('-size',162,'miniBatch size for DAE')
   cmd:option('-maxEpo',200,'maximum nb of epoch for DAE')
   cmd:option('-coefL1',0.1,'coef L1')
   cmd:option('coefL2',0.1,'coef L2')
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
class = {'1','2'}
-- This matrix records the current confusion across classes
trainConfusion = optim.ConfusionMatrix(classes)
validConfusion = optim.ConfusionMatrix(classes)
TvalidConfusion = optim.ConfusionMatrix(classes)
TtrainConfusion = optim.ConfusionMatrix(classes)
binaireConfusionT = optim.ConfusionMatrix(class)
binaireConfusionV = optim.ConfusionMatrix(class)
-------------------------------------------------
--   Data
-------------------------------------------------

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
--    Redimensionnement
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
---------------------------------------------------
--moyenneTensor=torch.CudaTensor(9,sourceTrainSet:size(3))
--stdTensor=torch.CudaTensor(9,sourceTrainSet:size(3))
--Rescal(sourceTrainSet,sourceValidSet,moyenneTensor,stdTensor)
--print('ok')

---------------------------------------------------
--    Data Augmentation Noise
---------------------------------------------------
print('==> generate Noise')
--generateNoise(sourceTrainSet,sourceTrainLabel,nTrain)
--generateNoise(targetTrainSet,targetTrainLabel,nTrain)


---------------------------------------------------
--    Data Augmentation 2
---------------------------------------------------
print ('==> Data Augmentation ')
--sourceTrainSet + sourceTrainLabel
posIn = 1 
pos = nSourceTrain/2 +1
while (posIn <= nSourceTrain/2) do
	dataAugmentationSujet2(sourceTrainSet,sourceTrainLabel,sourceTrainSet,sourceTrainLabel,posIn,pos,nTimeTrain,ntime)
	posIn = posIn + nTimeTrain/ntime
	pos = pos + nTimeTrain/ntime
end
print('Next 1/3')

--sourceValidSet +sourceValidLabel
posIn = 1
pos = nSourceValid/2 +1
while (posIn <= nSourceValid/2) do
        
dataAugmentationSujet2(sourceValidSet,sourceValidLabel,sourceValidSet,sourceValidLabel,posIn,pos,nTimeValid,ntime)
        posIn = posIn + nTimeValid/ntime
        pos = pos + nTimeValid/ntime
end

print('Next 2/3')

--targetTrainSet + targetTrainLabel

posIn = 1
pos = nTargetTrain/2 +1
while (posIn <= nTargetTrain/2) do
        dataAugmentationSujet2 
(targetTrainSet,targetTrainLabel,targetTrainSet,targetTrainLabel,posIn,pos,nTimeTrain,ntime)
        posIn = posIn + nTimeTrain/ntime
        pos = pos + nTimeTrain/ntime
end
print('Next 3/3')

--targetValidSet + targetValidLabel
posIn = 1
pos = nTargetValid/2 +1

while (posIn <= nTargetValid/2) do
        dataAugmentationSujet2 
(targetValidSet,targetValidLabel,targetValidSet,targetValidLabel,posIn,pos,nTimeValid,ntime)
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

---------------------------------------------------
-- 	Data Mix 
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

encoder = nn.Sequential()
noiser=nn.WhiteNoise(0,opt.sd)
encoder:add(noiser)
encoder:add(nn.SpatialConvolution(ntime,50 ,5, 3, 2, 2, 3, 3))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialConvolution(50,50 ,5, 3, 2, 2, 3, 3))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialAveragePooling(2,2,2,2))
encoder:cuda()

-----------------------------------------
--sampleSource=torch.CudaTensor(ntime,1,nFreq)
--for k=1,ntime do
--	sampleSource[k]:copy(sourceInputs[1][k])
--end
--feat = encoder:forward(sampleSource)
--print(feat:size())
--output = decoder:forward(feat)
--print(output:size())

--------------1-----------------
modelAv =nn.Sequential()
modelAv:add(nn.Reshape(50*2*13))
modelAv:add(nn.Linear(50*2*13,1000))
modelAv:add(nn.ReLU(true))
modelAv:add(nn.Linear(1000,600))
modelAv:add(nn.Dropout(0.8))
modelAv:add(nn.ReLU(true))
modelAv:cuda()

modelPred = nn.Sequential()
modelPred:add(nn.Linear(600,100))
modelPred:add(nn.Dropout(0.8))
modelPred:add(nn.ReLU(true))
modelPred:add(nn.Linear(100,9))
modelPred:add(nn.LogSoftMax())
modelPred:cuda()

modelClas = nn.Sequential()
modelClas:add(nn.GradientReversal())
modelClas:add(nn.Linear(600,300))
modelClas:add(nn.ReLU(true))
modelClas:add(nn.Linear(300,2))
--modelClas:add(nn.LogSoftMax())
modelClas:add(nn.Sigmoid())
modelClas:cuda()

--output1= modelAv:forward(feat)
--output2 = modelClas:forward(output1)
--print(output2)


--------------------------------------
-- Definition of the criterion
--------------------------------------

criterion = nn.ClassNLLCriterion()
--criterionClas = nn.ClassNLLCriterion()
criterionClas = nn.BCECriterion()
criterion:cuda()
criterionClas:cuda()
criterionDAE = nn.MSECriterion():cuda()

-- Retrieve the pointers to the parameters and gradParameters from the model for lat
featExtractorParams,featExtractorGradParams = modelAv:getParameters()
labelPredictorParams,labelPredictorGradParams = modelPred:getParameters()
domainClassifierParams,domainClassifierGradParams = modelClas:getParameters()
paramsEn,gradParamsEn = encoder:getParameters()
------------------------------------

paramsEn:cuda()
featExtractorParams:cuda()
featExtractorGradParams:cuda()
labelPredictorParams:cuda()

labelPredictorGradParams:cuda()
domainClassifierParams:cuda()
domainClassifierGradParams:cuda()
gradParamsEn:cuda()

-------------------------------------
-- Parametres DANN
params = torch.CudaTensor(featExtractorParams:size(1)+labelPredictorParams:size(1)+domainClassifierParams:size(1))
params:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
params:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
params:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)):copy(domainClassifierParams)

gradParams = torch.CudaTensor(featExtractorParams:size(1)+labelPredictorParams:size(1)+domainClassifierParams:size(1))
-------------------------------------

---------------------------------------
--   Learning function DANN
---------------------------------------
sourceInputs1 = torch.CudaTensor(opt.batchSize,ntime,1,nFreq)
targetInputs1 = torch.CudaTensor(opt.batchSize,ntime,1,nFreq)

function train()
print('==> train')
   local tick1 = sys.clock()
   

   shuffle = torch.randperm(sourceTrainSet:size(1))

   for t = 1,sourceTrainSet:size(1),opt.batchSize do
	  
          xlua.progress(t,sourceTrainSet:size(1))
          -- Define the minibatch
for mm=1,10 do

          for i = 1,opt.batchSize do
                 sourceInputs[i]:copy(sourceTrainSet[shuffle[t+i-1]])
                 sourceLabels[i] = sourceTrainLabel[shuffle[t+i-1]]
                 targetInputs[i]:copy(targetTrainSet[shuffle[t+i-1]])
                 targetLabels[i] = targetTrainLabel[shuffle[t+i-1]]
          end

	 sampleSource=torch.CudaTensor(ntime,1,nFreq)

	 for j=1, opt.batchSize do

		for m=1,ntime do
	        	sampleSource[m]:copy(sourceInputs[j][m])
		end
		sourceInputs1[j]:copy(sampleSource)
                for m=1,ntime do
                        sampleTarget[m]:copy(targetInputs[j][m])
                end
		targetInputs1[j]:copy(sampleTarget)
          end

	  local feval = function(x)
		 featExtractorParams:copy(x:narrow(1,1,featExtractorParams:size(1)))
		 labelPredictorParams:copy(x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)))
		 domainClassifierParams:copy(x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)))

		 featExtractorGradParams:zero()
		 labelPredictorGradParams:zero()
		 domainClassifierGradParams:zero()

		 sampleEn=encoder:forward(sourceInputs1)
		 feats = modelAv:forward(sampleEn)
		 preds = modelPred:forward(feats)

		 labelCost = criterion:forward(preds,sourceLabels)
		 labelDfdo = criterion:backward(preds, sourceLabels)

		 gradLabelPredictor = modelPred:backward(feats, labelDfdo)
		 modelAv:backward(sampleEn, gradLabelPredictor)
		 domPreds = modelClas:forward(feats)
		 domCost = criterionClas:forward(domPreds,torch.CudaTensor(domPreds:size(1),2):fill(0))

		 domDfdo = criterionClas:backward(domPreds,torch.CudaTensor(domPreds:size(1),2):fill(0))			
		 gradDomainClassifier = modelClas:backward(feats,domDfdo,opt.domainLambda) 
		 modelAv:backward(sampleEn,gradDomainClassifier,opt.domainLambda)

		 sampleTEn = encoder:forward(targetInputs1)
		 targetFeats = modelAv:forward(sampleTEn)

		 targetDomPreds = modelClas:forward(targetFeats)
		 targetDomCost = criterionClas:forward(targetDomPreds,torch.CudaTensor(targetDomPreds:size(1),2):fill(1))
		 targetDomDfdo = criterionClas:backward(targetDomPreds,torch.CudaTensor(targetDomPreds:size(1),2):fill(1))
		 targetGradDomainClassifier = modelClas:backward(targetFeats,targetDomDfdo,opt.domainLambda)  
		 modelAv:backward(sampleTEn,targetGradDomainClassifier,opt.domainLambda)
		 
		 params:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
		 params:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
		 params:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)):copy(domainClassifierParams)
		 
		 gradParams:narrow(1,1,featExtractorGradParams:size(1)):copy(featExtractorGradParams)
		 gradParams:narrow(1,featExtractorGradParams:size(1),labelPredictorGradParams:size(1)):copy(labelPredictorGradParams)
		 gradParams:narrow(1,featExtractorGradParams:size(1)+labelPredictorParams:size(1),domainClassifierGradParams:size(1)):copy(domainClassifierGradParams)
		 y=torch.sign(params)
		 gradParams:add(params:clone():mul(opt.coefL2))
		 return params,gradParams		 
	  end
	  optim.adam(feval,params,opt)
	end
   end
   print("tick" .. sys.clock()-tick1)
end

out = assert(io.open("./ResultsTrainEEG.csv", "w"))
out:write("Epoch     trainLoss     trainGlobalValid     validLoss     validGlobalValid     TestLoss     TestGlobal     LossSource     LossTarget")
out:write("\n")
out:write("\n")
prevLoss =10e12
splitter = ","

-----------------------------Data-Source-Target-Train------------------------------------------
sourceTrainInp = torch.CudaTensor(sourceTrainSet:size(1),sourceTrainSet:size(2),1,sourceTrainSet:size(3))
targetTrainInp = torch.CudaTensor(targetTrainSet:size(1),targetTrainSet:size(2),1,targetTrainSet:size(3))
sourceTrainLab = torch.CudaTensor(sourceTrainSet:size(1))
targetTrainLab = torch.CudaTensor(targetTrainSet:size(1))
sourceTrainLab:copy(sourceTrainLabel)
targetTrainLab:copy(targetTrainLabel)
sampleSource=torch.CudaTensor(sourceTrainSet:size(2),1,sourceTrainSet:size(3))
sampleTarget = torch.CudaTensor(targetTrainSet:size(2),1,targetTrainSet:size(3))
for i=1,sourceTrainSet:size(1) do 
	for m=1,ntime do
        	sampleSource[m]:copy(sourceTrainSet[i][m])
        end
	sourceTrainInp[i]:copy(sampleSource)
	for m=1,ntime do 
		sampleTarget[m]:copy(targetTrainSet[i][m])
	end
	targetTrainInp[i]:copy(sampleTarget)
end

----------------------------Data-Source-Target-Valid-----------------------------------------
sourceValidInp = torch.CudaTensor(sourceValidSet:size(1),sourceValidSet:size(2),1,sourceValidSet:size(3))
targetValidInp = torch.CudaTensor(targetValidSet:size(1),targetValidSet:size(2),1,targetValidSet:size(3))
sourceValidLab = torch.CudaTensor(sourceValidSet:size(1))
targetValidLab = torch.CudaTensor(targetValidSet:size(1))
sourceValidLab:copy(sourceValidLabel)
targetValidLab:copy(targetValidLabel)
sampleSource=torch.CudaTensor(sourceValidSet:size(2),1,sourceValidSet:size(3))
sampleTarget = torch.CudaTensor(targetValidSet:size(2),1,targetValidSet:size(3))
for i=1,sourceValidSet:size(1) do
        for m=1,ntime do
                sampleSource[m]:copy(sourceValidSet[i][m])
        end
	sourceValidInp[i]:copy(sampleSource)
        for m=1,ntime do
                sampleTarget[m]:copy(targetValidSet[i][m])
        end
	targetValidInp[i]:copy(sampleTarget)
end
---------------------------------------------------------
dataTrainInp = torch.CudaTensor(dataTrain:size(1),ntime,1,nFreq)
dataValidInp = torch.CudaTensor(dataValid:size(1),ntime,1,nFreq)

sampleSource=torch.CudaTensor(sourceValidSet:size(2),1,sourceValidSet:size(3))

for i=1,dataTrain:size(1) do
        for m=1,ntime do
                sampleSource[m]:copy(dataTrain[i][m])
        end
	dataTrainInp[i]:copy(sampleSource)
end

for i=1,dataValid:size(1) do
        for m=1,ntime do
                sampleSource[m]:copy(dataValid[i][m])
        end
	dataValidInp[i]:copy(sampleSource)
end


-------------------------------------------------------------------------------------------------

for i=1, opt.maxEpoch do 
	-- Evaluating the model
	------------------------------
	modelAv:evaluate()
	modelPred:evaluate()
	modelClas:evaluate()
	encoder:evaluate()
------------------------------ Test Train ------------------------------------------

	expEn = encoder:forward(sourceTrainInp)
	sourceFeats = modelAv:forward(expEn)
	sourceTrainPred = modelPred:forward(sourceFeats)
	sourceTrainLoss = criterion:forward(sourceTrainPred, sourceTrainLab)
	sourceDomPreds = modelClas:forward(sourceFeats)
	sourceDomCost = criterionClas:forward(sourceDomPreds,torch.CudaTensor(sourceDomPreds:size(1),2):fill(0))
---------------------------

	expTEn = encoder:forward(targetTrainInp)
	targetFeats = modelAv:forward(expTEn)
	targetDomPreds = modelClas:forward(targetFeats)
	targetDomCost = criterionClas:forward(targetDomPreds,torch.CudaTensor(targetDomPreds:size(1),2):fill(1))

	trainConfusion:batchAdd(sourceTrainPred, sourceTrainLab)

---------------------------------Test Source ---------------------------------------

        VexpEn = encoder:forward(sourceValidInp)
        VsourceFeats = modelAv:forward(VexpEn)
        VsourceTrainPred = modelPred:forward(VsourceFeats)
        VsourceTrainLoss = criterion:forward(VsourceTrainPred, sourceValidLab)
        VsourceDomPreds = modelClas:forward(VsourceFeats)
        VsourceDomCost = criterionClas:forward(VsourceDomPreds,torch.CudaTensor(VsourceDomPreds:size(1),2):fill(0))
---------------------------

        expTEn = encoder:forward(targetValidInp)
        VtargetFeats = modelAv:forward(expTEn)
        VtargetDomPreds = modelClas:forward(VtargetFeats)
        VtargetDomCost = criterionClas:forward(VtargetDomPreds,torch.CudaTensor(VtargetDomPreds:size(1),2):fill(1))

        validConfusion:batchAdd(VsourceTrainPred, sourceValidLab)

----------------------------------Test Target ------------------------------------
		
	TexpEn = encoder:forward(targetValidInp)
        targetFeats = modelAv:forward(TexpEn)
        targetValidPred = modelPred:forward(targetFeats)
        targetValidLoss = criterion:forward(targetValidPred, targetValidLab)

	TvalidConfusion:batchAdd(targetValidPred, targetValidLab)
------------------------------------------------------------------------
	-- Test Classification Train
		 
	Res = encoder:forward(dataTrainInp)
	featsR = modelAv:forward(Res)
	predsR = modelClas:forward(featsR)
	binaireConfusionT:batchAdd(predsR,dataLabelTrain)
	
------------------------------------------------------------------------
        -- Test Classification Valid

        ResV = encoder:forward(dataValidInp)
        featsV = modelAv:forward(ResV)
        predsV = modelClas:forward(featsV)
        binaireConfusionV:batchAdd(predsV,dataLabelValid)

	if i == opt.maxEpoch then
                matrix = assert(io.open("./MatrixConfusion.csv", "w"))
                for i=1, trainConfusion.mat:size(1) do
                        for j=1, trainConfusion.mat:size(2) do
                                matrix:write(trainConfusion.mat[i][j])
                                matrix:write(splitter)
                        end
                        matrix:write("\n")
                end
                matrix:close()

                matrix2 = assert(io.open("./MatrixConfusionValid.csv", "w"))
                for i=1, trainConfusion.mat:size(1) do
                        for j=1, trainConfusion.mat:size(2) do
                                matrix2:write(trainConfusion.mat[i][j])
                                matrix2:write(splitter)
                        end
                        matrix2:write("\n")
                end
                matrix2:close()

		matrix3 = assert(io.open("./MatrixConfusionTest.csv", "w"))
                for i=1, trainConfusion.mat:size(1) do
                        for j=1, TvalidConfusion.mat:size(2) do
                                matrix3:write(TvalidConfusion.mat[i][j])
                                matrix3:write(splitter)
                        end
                        matrix3:write("\n")
                end
                matrix3:close()
        end
	
	---------------------------------
	-- PRINT
	---------------------------------
	print("EPOCH: " .. i)

	print(trainConfusion)
	print(" + Train loss " .. sourceTrainLoss .. " " )

	print(validConfusion)
	print(" + Valid loss " .. VsourceTrainLoss .. " " )

	print(TvalidConfusion)
	print(" + Target Valid loss " .. targetValidLoss .. " " )

	print(binaireConfusionT)
	print(binaireConfusionV)
	---------------------------------
	out:write(i)
	out:write(splitter)
	out:write(sourceTrainLoss)
	out:write(splitter)
	out:write(trainConfusion.totalValid*100)
	out:write(splitter)
	out:write(VsourceTrainLoss)
	out:write(splitter)
	out:write(validConfusion.totalValid*100)
	out:write(splitter)
	out:write(targetValidLoss)
	out:write(splitter)
	out:write(TvalidConfusion.totalValid*100)
	out:write("\n")

   	trainConfusion:zero()
   	validConfusion:zero()
	TvalidConfusion:zero()
	binaireConfusionT:zero()
	binaireConfusionV:zero()

	modelAv:training()
	modelPred:training()
	modelClas:training()
	encoder:training()

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
	sourceTrainLoss =0
	VsourceTrainLoss =0
	targetValidLoss = 0
	train()

end

out:close()

