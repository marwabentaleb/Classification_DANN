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
   cmd:option('-maxEpoch', 150, 'maximum nb of epoch')
   cmd:option('-seed', 0, 'random seed')
   cmd:option('-domainLambda', 0.1, 'regularization term for transfer learning')
   cmd:option('-saveModel', false, 'flag for saving the model on disk at each epoch, if improvement')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-gpu', 1, 'nombre des Gpus')
   cmd:option('-sd',0.01,'noise')
   cmd:option('-maxEpo',200,'maximum nb of epoch for DAE')
   cmd:option('-coefL1',0.0001,'coef L1')
   cmd:option('-coefL2',0.01,'coef L2')
   cmd:option('-coef',2,'nombre d électrode')
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
		
		for i=0,(opt.coef-1) do 
			if i<10 then  
        			filename=string.format("./newdata/%s%u%s%u%s%s%u%s","S",s,"_D",indxDom,"_specgramCH","_0",i,".csv")
        			file=io.open(filename,"r") -- open file 
			end
			if i>9 then 
				filename=string.format("./newdata/%s%u%s%u%s%u%s","S",s,"_D",indxDom,"_specgramCH_",i,".csv")
                                file=io.open(filename,"r") -- open file 
			end 

        		frq=1 
        		for line in io.lines(filename) do -- line for time  
                		lineSplit=split(line,sep)
       	        		nLineSplit=table.getn(lineSplit)
				
			
                                	for time=1,nTimeTrain  do
                                        	trainTensor[s+i][time][frq] = lineSplit[time]
                                	end
                       
                               		for time=1,nTimeValid  do
                                        	validTensor[s+i][time][frq] = lineSplit[time+nTimeTrain]
                                	end
                        	frq=frq+1
                        
			end
			labelTrainTensor[s+i] = s
                	labelValidTensor[s+i] = s
			io.close(file)
		end
		s=s+1
	end
end  



function zeroMean(tensor,size,nTime)

	for n=1,tensor:size(1) do
        	for t=1,tensor:size(2) do
                	s=0
                	for freq=1,tensor:size(3) do
                        	s=s+tensor[n][t][freq]
                	end
                	for freq=1,tensor:size(3) do
                        	tensor[n][t][freq]=tensor[n][t][freq]-(s/tensor:size(3))
                	end
        	end
	end
end

function DataClas(tensorSource,tensorSourceLabel,tensorTarget,tensorTargetLabel,data,dataLabel,dataLabels)
        shuffleS = torch.randperm(tensorSource:size(1))
        shuffleT = torch.randperm(tensorTarget:size(1))
        p=1
	q=1
	for i=1, data:size(1) do
                if (i%2 == 0) then
                        data[i] = tensorSource[shuffleS[p]]
                        dataLabel[i] = 1
                        dataLabels[i] = tensorSourceLabel[shuffleS[p]]
                        p = p + 1
                else
                    	data[i] = tensorTarget[shuffleT[q]]
                        dataLabel[i] = 2
                        dataLabels[i] = tensorTargetLabel[shuffleT[q]]
                        q = q+1
                end
        end
end

function dataAugmentationSujet (trainTensor,labelTensor,tensor,nt)

	S=1
	T=1
	
	for s=1,(nt/ntime)*nSourceSujet*opt.coef do 
		for t=1,ntime do
			trainTensor[s][t]:copy(tensor[S][T])
			T=T+1
			if (T>tensor:size(2)) then 
				T=1
				S=S+1
			end
		end
		
		if (s<=((nt/ntime)*opt.coef)) then 
			labelTensor[s] = 1
		end 
		if ((nt/ntime*opt.coef)<s)and (s<=(2*nt/ntime*opt.coef)) then
                        labelTensor[s] = 2
                end
		if ((2*nt/ntime*opt.coef)<s)and(s<=(3*nt/ntime*opt.coef)) then
                        labelTensor[s] = 3
                end

		if ((3*nt/ntime*opt.coef)<s)and(s<=(4*nt/ntime*opt.coef)) then
                        labelTensor[s] = 4
                end
		
		if ((4*nt/ntime*opt.coef)<s)and(s<=(5*nt/ntime*opt.coef)) then
                        labelTensor[s] = 5
                end

		if ((5*nt/ntime*opt.coef)<s)and(s<=(6*nt/ntime*opt.coef)) then
                        labelTensor[s] = 6
                end

		if ((6*nt/ntime*opt.coef)<s)and(s<=(7*nt/ntime*opt.coef)) then
                        labelTensor[s] = 7
                end

		if ((7*nt/ntime*opt.coef)<s)and(s<=(8*nt/ntime*opt.coef)) then
                        labelTensor[s] = 8
                end

		if ((8*nt/ntime*opt.coef)<s)and(s<=(9*nt/ntime*opt.coef)) then
                        labelTensor[s] = 9
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
function alea(tensor, tensorLabel,tensor1, tensorLabel1, Size)
	shuffle = torch.randperm(Size)	

	aux=torch.CudaTensor(1,tensor:size(2),tensor:size(3))
	aux1=torch.CudaTensor(1,tensor:size(2),tensor:size(3))

	for i=1, Size do 
		aux:copy(tensor[shuffle[i]])
		tensor[shuffle[i]]:copy(tensor[i])
		tensor[i]:copy(aux)
		auxLabel=tensorLabel[i]
		tensorLabel[i]=tensorLabel[shuffle[i]]
		tensorLabel[shuffle[i]]=auxLabel

		aux1:copy(tensor1[shuffle[i]])
		tensor1[shuffle[i]]:copy(tensor1[i])
		tensor1[i]:copy(aux1)
		auxLabel1=tensorLabel1[i]
		tensorLabel1[i]=tensorLabel1[shuffle[i]]
		tensorLabel1[shuffle[i]]=auxLabel1
	end


end

sep=','
-------------------------------------------------
--   Data
-------------------------------------------------

-- Source Domain
nSourceSujet = 9 
nTimeTrain = 2400 
nTimeValid = 600
nFreq = 101
ntime=10
nTime=3000
coef = 2
nSourceTrain = 2*(nTimeTrain/ntime)*nSourceSujet*opt.coef
nSourceValid = 2*(nTimeValid/ntime)*nSourceSujet*opt.coef

sourceTrainSet = torch.CudaTensor(nSourceTrain,ntime,nFreq):zero()
sourceTrainLabel = torch.CudaTensor(nSourceTrain):zero()
sourceValidSet = torch.CudaTensor(nSourceValid,ntime,nFreq):zero()
sourceValidLabel = torch.CudaTensor(nSourceValid):zero()

sourceT = torch.CudaTensor(opt.coef*nSourceSujet,nTimeTrain,nFreq):zero()
sourceTLabels = torch.CudaTensor(opt.coef*nSourceSujet):zero()
sourceV = torch.CudaTensor(opt.coef*nSourceSujet,nTimeValid,nFreq):zero()
sourceVLabels = torch.CudaTensor(opt.coef*nSourceSujet):zero()


sourceInputs = torch.CudaTensor(opt.batchSize,sourceTrainSet:size(2),sourceTrainSet:size(3)):zero()
sourceLabels = torch.CudaTensor(opt.batchSize):zero()

-- Target Domain
nTargetSujet = 9

nTargetTrain = 2*(nTimeTrain/ntime)*nTargetSujet*opt.coef
nTargetValid = 2*(nTimeValid/ntime)*nTargetSujet*opt.coef

targetTrainSet = torch.CudaTensor(nTargetTrain,ntime,nFreq):zero()
targetTrainLabel = torch.CudaTensor(nTargetTrain):zero()
targetValidSet = torch.CudaTensor(nTargetValid,ntime,nFreq):zero()
targetValidLabel = torch.CudaTensor(nTargetValid):zero()

targetT = torch.CudaTensor(opt.coef*nTargetSujet,nTimeTrain,nFreq):zero()
targetTLabels = torch.CudaTensor(opt.coef*nTargetSujet):zero()
targetV = torch.CudaTensor(opt.coef*nTargetSujet,nTimeValid,nFreq):zero()
targetVLabels= torch.CudaTensor(opt.coef*nTargetSujet):zero()


targetInputs = torch.CudaTensor(opt.batchSize,targetTrainSet:size(2),targetTrainSet:size(3)):zero()
targetLabels = torch.CudaTensor(opt.batchSize):zero()

print('==> Generate function')

Generate(sourceT,sourceTLabels,sourceV,sourceVLabels,2,nSourceSujet)
Generate(targetT,targetTLabels,targetV,targetVLabels,4,nTargetSujet)

---------------------------------------------------
--    Zero Mean
---------------------------------------------------
print('==> zeroMean function')

-- sourceTrainSet
zeroMean(sourceT,opt.coef*nSourceSujet,sourceT:size(2))
print('Next 1/3')
-- targetTrainSet
zeroMean(targetT,opt.coef*nTargetSujet,targetT:size(2))
print('Next 2/3')
-- sourceValid
zeroMean(sourceV,opt.coef*nSourceSujet,sourceV:size(2))
print('Next 3/3')
--targetValid
zeroMean(targetV,opt.coef*nTargetSujet,targetV:size(2))
print('ok')
---------------------------------------------------
--    Redimensionnement
---------------------------------------------------
print('==> Redimensionnement')

-- sourceTrainSet + sourceTrainLabel 
dataAugmentationSujet (sourceTrainSet,sourceTrainLabel,sourceT,nTimeTrain)
print('Next 1/3')

-- sourceValidSet +sourceValidLabel
dataAugmentationSujet (sourceValidSet,sourceValidLabel,sourceV,nTimeValid)
print('Next 2/3')

-- targetTrainSet + targetTrainLabel
dataAugmentationSujet (targetTrainSet,targetTrainLabel,targetT,nTimeTrain)
print('Next 3/3')

-- targetValidSet + targetValidLabel
dataAugmentationSujet (targetValidSet,targetValidLabel,targetV,nTimeValid)
print(sourceTrainLabel)
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
print(sourceTrainLabel)
---------------------------------------------------
---     Alea
---------------------------------------------------
print('==>Alea')
alea(sourceTrainSet,sourceTrainLabel,targetTrainSet,targetTrainLabel,nSourceTrain)
print('Next')
alea(sourceValidSet,sourceValidLabel,targetValidSet,targetValidLabel,nSourceValid)
---------------------------------------------------
print('==> data')
data = torch.CudaTensor(sourceTrainSet:size(1) + targetTrainSet:size(2),ntime,nFreq):zero()
dataLabel = torch.ones(sourceTrainSet:size(1) + targetTrainSet:size(2)):zero()
dataLabel:cuda()
DataClas(sourceTrainSet,targetTrainSet,data,dataLabel)
print('==> aleaData')
aleaData(data,dataLabel)

---------------------------------------------------
--      Data Mix 
---------------------------------------------------
print('==> data')

dataTrain = torch.CudaTensor(sourceTrainSet:size(1)+targetTrainSet:size(1),ntime,nFreq)
dataLabelTrain = torch.CudaTensor(sourceTrainSet:size(1)+targetTrainSet:size(1))
dataLabelsT = torch.CudaTensor(sourceTrainSet:size(1)+targetTrainSet:size(1))
--DataClas(sourceTrainSet,sourceTrainLabel,targetTrainSet,targetTrainLabel,dataTrain,dataLabelTrain,dataLabelsT)
--aleaData(dataTrain,dataLabelTrain)

dataValid = torch.CudaTensor(sourceValidSet:size(1)+targetValidSet:size(1),ntime,nFreq)
dataLabelValid = torch.CudaTensor(sourceValidSet:size(1)+targetValidSet:size(1))
dataLabelsV = torch.CudaTensor(sourceValidSet:size(1) + targetValidSet:size(1))
--DataClas(sourceValidSet,sourceValidLabel,targetValidSet,targetValidLabel,dataValid,dataLabelValid,dataLabelsV)
---------------------------------------------------
--   Definition of the model 
---------------------------------------------------
modelAv =nn.Sequential()
noiser=nn.WhiteNoise(0,opt.sd)
modelAv:add(noiser)
modelAv:add(nn.SpatialConvolution(ntime,50 ,5, 3, 2, 2, 3, 3))
modelAv:add(nn.ReLU(true))
modelAv:add(nn.SpatialConvolution(50,50 ,5, 3, 2, 2, 3, 3))
modelAv:add(nn.ReLU(true))
modelAv:add(nn.SpatialAveragePooling(2,2,2,2))

--------------1-----------------
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
--modelClas:add(nn.Dropout(0.4))
modelClas:add(nn.ReLU(true))
modelClas:add(nn.Linear(300,2))
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
------------------------------------

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
sourceInputs1 = torch.CudaTensor(opt.batchSize,ntime,1,nFreq)
targetInputs1 = torch.CudaTensor(opt.batchSize,ntime,1,nFreq)

function train()
print('==> train')
   local tick1 = sys.clock()
   

   shuffle = torch.randperm(sourceTrainSet:size(1))
   
   for t = 1,sourceTrainSet:size(1),opt.batchSize do
	  
	  xlua.progress(t,sourceTrainSet:size(1))
	  
	  -- Define the minibatch
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
for m=1,5 do 
	  local feval = function(x)
		 featExtractorParams:copy(x:narrow(1,1,featExtractorParams:size(1)))
		 labelPredictorParams:copy(x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)))
		 domainClassifierParams:copy(x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)))

		 featExtractorGradParams:zero()
		 labelPredictorGradParams:zero()
		 domainClassifierGradParams:zero()
		feats = modelAv:forward(sourceInputs1)
                 preds = modelPred:forward(feats)

                 labelCost = criterion:forward(preds,sourceLabels)
                 labelDfdo = criterion:backward(preds, sourceLabels)

                 gradLabelPredictor = modelPred:backward(feats, labelDfdo)
                 modelAv:backward(sourceInputs1, gradLabelPredictor)

                 domPreds = modelClas:forward(feats)
                 domCost = criterionClas:forward(domPreds,torch.CudaTensor(domPreds:size(1),2):fill(0))

                 domDfdo = criterionClas:backward(domPreds,torch.CudaTensor(domPreds:size(1),2):fill(0))
                 gradDomainClassifier = modelClas:backward(feats,domDfdo,opt.domainLambda)
                 modelAv:backward(sourceInputs1,gradDomainClassifier,opt.domainLambda)

			 --- Target propagation
		targetFeats = modelAv:forward(targetInputs1)

                 targetDomPreds = modelClas:forward(targetFeats)
                 targetDomCost = criterionClas:forward(targetDomPreds,torch.CudaTensor(targetDomPreds:size(1),2):fill(1))
                 targetDomDfdo = criterionClas:backward(targetDomPreds,torch.CudaTensor(targetDomPreds:size(1),2):fill(1))
                 targetGradDomainClassifier = modelClas:backward(targetFeats,targetDomDfdo,opt.domainLambda)
                 modelAv:backward(targetInputs1,targetGradDomainClassifier,opt.domainLambda)

                 params:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
                 params:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
                 params:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)):copy(domainClassifierParams)

                 gradParams:narrow(1,1,featExtractorGradParams:size(1)):copy(featExtractorGradParams)
                 gradParams:narrow(1,featExtractorGradParams:size(1),labelPredictorGradParams:size(1)):copy(labelPredictorGradParams)
                 gradParams:narrow(1,featExtractorGradParams:size(1)+labelPredictorParams:size(1),domainClassifierGradParams:size(1)):copy(domainClassifierGradParams)
                 --y=torch.sign(params)
                 gradParams:add(params:clone():mul(opt.coefL2))
                 --gradParams:add(y:mul(opt.coefL1))

		 return params,gradParams		 
       	   end
	  optim.adam(feval,params,opt)
end   
end
   print("tick" .. sys.clock()-tick1)
end

out = assert(io.open("./ResultsTrainEEG.csv", "w"))
out:write("Epoch     trainLoss     trainGlobal     validLoss     validGlobal     targetLoss     targetGlobal")
out:write("\n")
out:write("\n")
prevLoss =10e12
splitter = ","

-----------------------------Data-Source-Target-Train------------------------------------------
sourceTrainInp = 
torch.CudaTensor(sourceTrainSet:size(1),sourceTrainSet:size(2),1,sourceTrainSet:size(3))
targetTrainInp = 
torch.CudaTensor(targetTrainSet:size(1),targetTrainSet:size(2),1,targetTrainSet:size(3))
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

sourceValidInp = 
torch.CudaTensor(sourceValidSet:size(1),sourceValidSet:size(2),1,sourceValidSet:size(3))
targetValidInp = 
torch.CudaTensor(targetValidSet:size(1),targetValidSet:size(2),1,targetValidSet:size(3))
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
cord = assert(io.open("./Cord.csv", "w"))
cord:write("X   Y       class")
cord:write("/n")
cord:write("/n")

cordBin = assert(io.open("./CordBin.csv", "w"))
cordBin:write("X   Y       class")
cordBin:write("/n")
cordBin:write("/n")
for i=1, opt.maxEpoch do
        -- Evaluating the model
        ------------------------------
        modelAv:evaluate()
        modelPred:evaluate()
        modelClas:evaluate()

------------------------------ Test Train ------------------------------------------

        --expEn = encoder:forward(sourceTrainInp)
        sourceFeats = modelAv:forward(sourceTrainInp)
	sourceTrainPred = modelPred:forward(sourceFeats)
	sourceTrainLoss = criterion:forward(sourceTrainPred, sourceTrainLab)
        sourceDomPreds = modelClas:forward(sourceFeats)
        sourceDomCost = criterionClas:forward(sourceDomPreds,torch.CudaTensor(sourceDomPreds:size(1),2):fill(0))

        --if i%opt.maxEpoch == 0 then 
        --sourceFeats = Cord:forward(sourceFeats)

        --      for k=1, sourceFeats:size(1) do
        --              cord:write(sourceFeats[k][1])
        --              cord:write(splitter)
        --              cord:write(sourceFeats[k][2])
        --              cord:write(splitter)
        --              min= torch.min(sourceTrainPred[k])
        --              ind=1
        --              for d=1,sourceTrainPred[k]:size(1) do 
        --                      if sourceTrainPred[k][d] == min then ind = d end
        --              end
        --              --cord:write(SourceTrainLabel[k])
        --              cord:write(ind)
        --              cord:write('\n')
        --      end
        --end

---------------------------
---------------------------

        --expTEn = encoder:forward(targetTrainInp)
        targetFeats = modelAv:forward(targetTrainInp)
        targetDomPreds = modelClas:forward(targetFeats)
        targetDomCost = criterionClas:forward(targetDomPreds,torch.CudaTensor(targetDomPreds:size(1),2):fill(1))
        trainConfusion:batchAdd(sourceTrainPred, sourceTrainLab)

---------------------------------Test Source ---------------------------------------

        --VexpEn = encoder:forward(sourceValidInp)
        VsourceFeats = modelAv:forward(sourceValidInp)
        VsourceTrainPred = modelPred:forward(VsourceFeats)
        VsourceTrainLoss = criterion:forward(VsourceTrainPred, sourceValidLab)
        VsourceDomPreds = modelClas:forward(VsourceFeats)
        VsourceDomCost = criterionClas:forward(VsourceDomPreds,torch.CudaTensor(VsourceDomPreds:size(1),2):fill(0))
---------------------------

        --expTEn = encoder:forward(targetValidInp)
        VtargetFeats = modelAv:forward(targetValidInp)
        VtargetDomPreds = modelClas:forward(VtargetFeats)
        VtargetDomCost = criterionClas:forward(VtargetDomPreds,torch.CudaTensor(VtargetDomPreds:size(1),2):fill(1))

        validConfusion:batchAdd(VsourceTrainPred, sourceValidLab)
----------------------------------Test Target ------------------------------------

        --TexpEn = encoder:forward(targetValidInp)
        targetFeats = modelAv:forward(targetValidInp)
        targetValidPred = modelPred:forward(targetFeats)
        targetValidLoss = criterion:forward(targetValidPred, targetValidLab)

        TvalidConfusion:batchAdd(targetValidPred, targetValidLab)
	print('ok')
------------------------------------------------------------------------
        -- Test Classification Train

--      Res = encoder:forward(dataTrainInp)
      --  featsR = modelAv:forward(dataTrainInp)
       -- predsR = modelClas:forward(featsR)
       -- binaireConfusionT:batchAdd(predsR,dataLabelTrain)

------------------------------------------------------------------------
        -- Test Classification Valid

        --ResV = encoder:forward(dataValidInp)
       -- featsV = modelAv:forward(dataValidInp)
       -- predsV = modelClas:forward(featsV)

        --binaireConfusionV:batchAdd(predsV,dataLabelValid)

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

        --print(binaireConfusionT)
        --print(binaireConfusionV)
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

cord:close()

