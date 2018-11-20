package gradient.optimumDetection;

import java.util.Arrays;

import function.common.benchmark.RosenbrockBananaFunction;
import function.decorators.DerivativeFunctionCallCounterWrapper;
import gradient.GradientDescent;
import gradient.updater.AdamGradientUpdater;
import gradient.updater.GradientClipperUpdater;
import gradient.updater.GradientEnhancerUpdater;
import gradient.updater.IGradientBasedUpdater;
import optimization.algorithm.IOptimizationAlgorithm;
import optimization.algorithm.decorator.TimedOptimizationAlgorithm;
import optimization.decoder.IDecoder;
import optimization.decoder.PassThroughDoubleDecoder;
import optimization.encoder.IEncoder;
import optimization.encoder.PassThroughEncoder;
import optimization.fittnesEvaluator.FunctionValueFitnessEvaluator;
import optimization.fittnesEvaluator.ThroughOneFitnessEvaluator;
import optimization.fittnesEvaluator.observable.Best2DUnitGraphngObserver;
import optimization.fittnesEvaluator.observable.BestObserver;
import optimization.fittnesEvaluator.observable.PerChromosomeObservableFitnessEvaluator;
import optimization.fittnesEvaluator.observable.PrintBestObserver;
import optimization.fittnesEvaluator.observable.SleepOnBestObserver;
import optimization.solution.DoubleArraySolution;
import optimization.startSolutionGenerator.RandomStartSolutionGenerator;
import optimization.stopper.CompositeOptimisationStopper;
import optimization.stopper.FunctionValueStopper;
import optimization.stopper.GenerationNumberEvolutionStopper;
import optimization.stopper.IOptimisationStopper;
import optimization.utility.AlgorithmsPresentationUtility;
import ui.graph.SimpleGraph;

public class BananaFunctionParameterDetection {

	private BananaFunctionParameterDetection() {}
	
	
	public static void main(String[] args) {
		
		double acceptableErrorRate = 10e-9;
		int maximumNumberOfGenerations = 50_000_000;
		
		//Function to optimize
		RosenbrockBananaFunction function = new RosenbrockBananaFunction();
		int variableCount = function.getVariableCount();
		double[] min = function.getStandardSearchMin();
		double[] max = function.getStandardSearchMax();
		DerivativeFunctionCallCounterWrapper wrappedFunction = new DerivativeFunctionCallCounterWrapper(function);		
		
		//Start UI
		SimpleGraph graph = new SimpleGraph(10,10);
		graph.display();
		
		//Decoder
		IDecoder<DoubleArraySolution,double[]> decoder = new PassThroughDoubleDecoder();

		//Encoder
		IEncoder<DoubleArraySolution> encoder = new PassThroughEncoder();

		//Start solution generator
		RandomStartSolutionGenerator startSolutionGenerator = new RandomStartSolutionGenerator(variableCount, min, max);

		
		//Fitness evaluator
		PerChromosomeObservableFitnessEvaluator<DoubleArraySolution> evaluator = new PerChromosomeObservableFitnessEvaluator<>(v->
			 ThroughOneFitnessEvaluator.evaluationMethod.applyAsDouble(
						FunctionValueFitnessEvaluator.evaluationMethod.applyAsDouble(v)
		));
		int sleepTimeInMs = 50;
		evaluator.addObserver(new BestObserver<>(decoder, Arrays.asList(
				new PrintBestObserver<>(System.out),
				new Best2DUnitGraphngObserver<>(graph),
				//TODO: include to watch step by step progress
				new SleepOnBestObserver<>(sleepTimeInMs)
		)));

		//Updater
		double learningRate = 0.05;
		IGradientBasedUpdater updater = new GradientEnhancerUpdater(
				new GradientClipperUpdater(new AdamGradientUpdater(learningRate,variableCount),1), 25, 1e-3);
		
		//Optimization stopper
		IOptimisationStopper<DoubleArraySolution> stopper = new CompositeOptimisationStopper<>(Arrays.asList(
			new FunctionValueStopper<>(acceptableErrorRate),
			new GenerationNumberEvolutionStopper<>(maximumNumberOfGenerations)
		));
		
		//Optimization
		IOptimizationAlgorithm<DoubleArraySolution> optimizationAlgorithm = 
				new GradientDescent<DoubleArraySolution>(decoder, encoder , evaluator, updater, stopper, wrappedFunction, startSolutionGenerator);
				
		TimedOptimizationAlgorithm<DoubleArraySolution> timedOptAlgorithm = new TimedOptimizationAlgorithm<>(optimizationAlgorithm);
		
		//Solution presentation
		DoubleArraySolution solution = timedOptAlgorithm.run();
		System.out.println();
		AlgorithmsPresentationUtility.printExecutionTime(timedOptAlgorithm.getExecutionTime());
		System.out.println("Solution: "  + solution);
		System.out.println("Optimum solution: " + Arrays.toString(function.getMinValueCoordinates()));
		System.out.println("Value: " + wrappedFunction.applyAsDouble(solution.values) + "\n");
		AlgorithmsPresentationUtility.printEvaluationCount(wrappedFunction.getEvaluationCount());
	}
	
}
