    %% SETUP

% Environment model
mdl = "WaterTank_system";   % Your Simulink model name
blk = mdl + "/RL Agent";    % Path to the RL Agent block

% Load Simulink model
open_system(mdl)

%% 1. DEFINE OBSERVATION AND ACTION INFO

obsInfo = rlNumericSpec([1 1], ...
    'LowerLimit', -100, ...
    'UpperLimit', 100);
obsInfo.Name = "error_only";

actInfo = rlNumericSpec([1 1], ...
    'LowerLimit', 0, ...
    'UpperLimit', 1);
actInfo.Name = "pump_flow";

env = rlSimulinkEnv(mdl, blk, obsInfo, actInfo);

%% 2. CREATE CRITIC NETWORK

obsPath = featureInputLayer(1, Name="obsInLyr");
actPath = featureInputLayer(1, Name="actInLyr");

commonPath = [
    concatenationLayer(1,2,Name="concat")
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(1, Name="QValue")
];

criticNet = dlnetwork();
criticNet = addLayers(criticNet, obsPath);
criticNet = addLayers(criticNet, actPath);
criticNet = addLayers(criticNet, commonPath);
criticNet = connectLayers(criticNet, "obsInLyr", "concat/in1");
criticNet = connectLayers(criticNet, "actInLyr", "concat/in2");

critic = rlQValueFunction(criticNet, obsInfo, actInfo, ...
    ObservationInputNames="obsInLyr", ...
    ActionInputNames="actInLyr");

%% 3. CREATE ACTOR NETWORK

actorNet = [
    featureInputLayer(1, Name="obsInput")
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(1)
    tanhLayer
    scalingLayer('Scale', 1) 
];

actorNet = dlnetwork(actorNet);

actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo, ...
    ObservationInputNames="obsInput");

%% 4. CREATE AGENT

agentOpts = rlDDPGAgentOptions(...
    SampleTime = 1.0, ...
    DiscountFactor = 0.99, ...
    MiniBatchSize = 64, ...
    ExperienceBufferLength = 1e6);

agentOpts.ActorOptimizerOptions.LearnRate = 5e-5;
agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;
agentOpts.NoiseOptions.StandardDeviation = 0.1;
agentOpts.NoiseOptions.StandardDeviationDecayRate = 1e-5;

agent = rlDDPGAgent(actor, critic, agentOpts);

%% 5. TRAINING OPTIONS

trainOpts = rlTrainingOptions(...
    MaxEpisodes=1000, ...
    MaxStepsPerEpisode=200, ...
    Plots="training-progress", ...
    Verbose=false, ...
    StopTrainingCriteria="AverageReward", ...
    StopTrainingValue=2000);

%% 6. TRAIN OR LOAD

doTraining = true;

if doTraining
    trainingStats = train(agent, env, trainOpts);
    save("trainedSoilMoistureAgent.mat", "agent");
else
    load("trainedSoilMoistureAgent.mat", "agent");
end