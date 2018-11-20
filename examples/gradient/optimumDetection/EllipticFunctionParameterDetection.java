package gradient.optimumDetection;

import java.util.Arrays;

import function.IFunction;
import function.common.ElipticFunction;
import function.decorators.DerivativeFunctionCallCounterWrapper;
import gradient.GradientDescent;
import gradient.updater.IGradientBasedUpdater;
import gradient.updater.MomentGradientUpdater;
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
import optimization.startSolutionGenerator.IStartSolutionGenerator;
import optimization.startSolutionGenerator.RandomStartSolutionGenerator;
import optimization.stopper.CompositeOptimisationStopper;
import optimization.stopper.FunctionValueStopper;
import optimization.stopper.GenerationNumberEvolutionStopper;
import optimization.stopper.IOptimisationStopper;
import optimization.utility.AlgorithmsPresentationUtility;
import ui.graph.SimpleGraph;

public class EllipticFunctionParameterDetection {

	private EllipticFunctionParameterDetection() {}
	
	
	public static void main(String[] args) {
		
		double acceptableErrorRate = 10e-6;
		int maximumNumberOfGenerations = 1_000_000;
		
		//Function to optimize
		IFunction<double[]> elipticFunction = new ElipticFunction();
		int variableCount = elipticFunction.getVariableCount();
		DerivativeFunctionCallCounterWrapper function = new DerivativeFunctionCallCounterWrapper(elipticFunction);		
		
		//Start UI
		SimpleGraph graph = new SimpleGraph(12,12);
		graph.display();

		//Start solution generator
		IStartSolutionGenerator<DoubleArraySolution> solutionGenerator = new RandomStartSolutionGenerator(variableCount, -10, 10);
		
		//Decoder
		IDecoder<DoubleArraySolution,double[]> decoder = new PassThroughDoubleDecoder();

		//Encoder
		IEncoder<DoubleArraySolution> encoder = new PassThroughEncoder();

		//Fitness evaluator
		PerChromosomeObservableFitnessEvaluator<DoubleArraySolution> evaluator = new PerChromosomeObservableFitnessEvaluator<>(v->
			 ThroughOneFitnessEvaluator.evaluationMethod.applyAsDouble(
						FunctionValueFitnessEvaluator.evaluationMethod.applyAsDouble(v)
		));
		int sleepTimeInMs = 50;
		evaluator.addObserver(new BestObserver<>(decoder, Arrays.asList(
				new PrintBestObserver<DoubleArraySolution,double[]>(System.out)
				,
				new Best2DUnitGraphngObserver<>(graph)
				,
				//TODO: include to watch step by step progress
				new SleepOnBestObserver<DoubleArraySolution,double[]>(sleepTimeInMs)
		)));

		//Updater
		double learningRate = 0.05;
		double momentDecreaseFactor = 0.6;
		IGradientBasedUpdater updater = new MomentGradientUpdater(learningRate, momentDecreaseFactor, variableCount);
		
		//Optimization stopper
		IOptimisationStopper<DoubleArraySolution> stopper = new CompositeOptimisationStopper<>(Arrays.asList(
			new FunctionValueStopper<>(acceptableErrorRate),
			new GenerationNumberEvolutionStopper<>(maximumNumberOfGenerations)
		));
		
		//Optimization
		IOptimizationAlgorithm<DoubleArraySolution> optimizationAlgorithm = 
				new GradientDescent<DoubleArraySolution>(decoder, encoder , evaluator, updater, stopper, function, solutionGenerator);
				
		TimedOptimizationAlgorithm<DoubleArraySolution> timedOptAlgorithm = new TimedOptimizationAlgorithm<>(optimizationAlgorithm);
		
		//Solution presentation
		DoubleArraySolution solution = timedOptAlgorithm.run();
		System.out.println();
		AlgorithmsPresentationUtility.printExecutionTime(timedOptAlgorithm.getExecutionTime());
		System.out.println("Solution: "  + solution);
		System.out.println("Value: " + function.applyAsDouble(solution.getValues()));
		AlgorithmsPresentationUtility.printEvaluationCount(function.getEvaluationCount());
	}
	
}
