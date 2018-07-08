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
   cmd:option('-learningRate', 0.00000001, 'learning rate at t=0') -- ou 0.0000001
   cmd:option('-rateDAE',0.001,'learning rate for DAE at t=0')
   cmd:option('-batchSize', 9, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-maxEpoch', 150, 'maximum nb of epoch')
   cmd:option('-seed', 0, 'random seed')
   cmd:option('-domainLambda', 0.1, 'regularization term for transfer learning')
   cmd:option('-saveModel', false, 'flag for saving the model on disk at each epoch, if improvement')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-gpu', 1, 'nombre des Gpus')
   cmd:option('-sd',0.01,'noise')
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
TvalidConfusion = optim.ConfusionMatrix(classes)
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

        	frq=1 
        	for line in io.lines(filename) do -- line for time  
                	lineSplit=split(line,sep)
       	        	nLineSplit=table.getn(lineSplit)
			
                                for time=1,nTimeTrain  do
                                        trainTensor[s][time][frq] = lineSplit[time]
                                end
                       
                                for time=1,nTimeValid  do
                                        validTensor[s][time][frq] = lineSplit[time+nTimeTrain]
                                end
                        frq=frq+1
                        
		end
		labelTrainTensor[s] = s
                labelValidTensor[s] = s
		
        	s=s+1
		io.close(file)
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
------------------------------------------------
-- Source Domain
nSourceSujet = 9 
nTimeTrain = 460
nTimeValid = 120
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

---------------------------------------------------
--    Data Augmentation 2
---------------------------------------------------
print ('==> Data Augmentation 2')
--sourceTrainSet + sourceTrainLabel
posIn = 1 
pos = nSourceTrain/2 +1
while (posIn <= nSourceTrain/2) do
	dataAugmentationSujet2 
(sourceTrainSet,sourceTrainLabel,sourceTrainSet,sourceTrainLabel,posIn,pos,nTimeTrain,ntime)
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
--     Alea
---------------------------------------------------
print('==>Alea')
alea(sourceTrainSet,sourceTrainLabel,targetTrainSet,targetTrainLabel,nSourceTrain)
print('Next')
alea(sourceValidSet,sourceValidLabel,targetValidSet,targetValidLabel,nSourceValid)


---------------------------------------------------
--   Definition of the model 
---------------------------------------------------

print('==> model')

modelAv = nn.Sequential()
modelAv:add(nn.Reshape(1*ntime*nFreq))
modelAv:add(nn.Linear(1*ntime*nFreq,1800))
modelAv:add(nn.ReLU(true))
modelAv:add(nn.Linear(1800,1000))
modelAv:add(nn.Dropout(0.7))
modelAv:add(nn.ReLU(true))
modelAv:add(nn.Linear(1000,600))
modelAv:add(nn.ReLU(true))
modelAv:cuda()

modelPred = nn.Sequential()
modelPred:add(nn.Linear(600,100))
modelPred:add(nn.Dropout(0.6))
modelPred:add(nn.Linear(100,9))
modelPred:add(nn.LogSoftMax())
modelPred:cuda()

---------------------------------------
--------------------------------------
-- Definition of the criterion
--------------------------------------

criterion = nn.ClassNLLCriterion()
criterion:cuda()

-- Retrieve the pointers to the parameters and gradParameters from the model for lat
params,gradParams = modelAv:getParameters()

------------------------------------
params:cuda()
gradParams:cuda()

----------------------------------------
--   Learning function
---------------------------------------

function train()

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


	  local feval = function(x)
		 params:copy(x)

		 gradParams:zero()

		 for j=1, opt.batchSize do
			 sampleSource=torch.CudaTensor(1,ntime,nFreq)
		         sampleSource:copy(sourceInputs[j])

			 feats = modelAv:forward(sampleSource)
			 preds = modelPred:forward(feats)

			 labelCost = criterion:forward(preds,sourceLabels[j])

			 labelDfdo = criterion:backward(preds, sourceLabels[j])

			 gradLabelPredictor = modelPred:backward(feats, labelDfdo)
			 modelAv:backward(sampleSource, gradLabelPredictor)

 		 end
		 return params,gradParams		 
	  end
	  optim.adam(feval,params,opt)

   end
   print("tick" .. sys.clock()-tick1)
end


out = assert(io.open("./ResultsTrainEEG.csv", "w"))
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

	tensorConfusion=torch.zeros(sourceTrainSet:size(1),9):cuda()

	-------------------------------
	trainLoss = 0
	for i=1,sourceTrainSet:size(1) do 
		expSource=torch.CudaTensor(1,ntime,nFreq)
		expSource:copy(sourceTrainSet[i])

		sourceFeats = modelAv:forward(expSource)
		sourceTrainPred = modelPred:forward(sourceFeats)
		sourceTrainLoss = criterion:forward(sourceTrainPred, sourceTrainLabel[i])

		tensorConfusion[i]:copy(sourceTrainPred)
		trainLoss=trainLoss+sourceTrainLoss
	end



	trainLoss = trainLoss/ sourceTrainSet:size(1)
	trainConfusion:batchAdd(tensorConfusion, sourceTrainLabel)

	tensorConfusion2 = torch.zeros(sourceValidSet:size(1),9):cuda()
	validLoss=0
	for j=1,sourceValidSet:size(1) do
		expValid=torch.CudaTensor(1,ntime,nFreq)
		expValid:copy(sourceValidSet[j])

		validPred = modelPred:forward(modelAv:forward(expValid))
		validLoss = criterion:forward(validPred, sourceValidLabel[j])

		tensorConfusion2[j]:copy(validPred)
		validLoss=validLoss + validLoss
	end
	validLoss = validLoss/sourceValidSet:size(1)

	validConfusion:batchAdd(tensorConfusion2, sourceValidLabel)
	
------------------------------------------------------------------------


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
        end
	
	---------------------------------
	-- PRINT
	---------------------------------
	print("EPOCH: " .. i)

	print(trainConfusion)
	print(" + Train loss " .. trainLoss .. " " )

	print(validConfusion)
	print(" + Valid loss " .. validLoss .. " " )

	---------------------------------
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

   	trainConfusion:zero()
   	validConfusion:zero()
	
	modelAv:training()
	modelPred:training()

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

	train()

end

out:close()

