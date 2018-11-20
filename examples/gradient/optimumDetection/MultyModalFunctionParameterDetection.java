package gradient.optimumDetection;

import java.awt.Color;
import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

import function.IFunction;
import function.decorators.DerivativeFunctionCallCounterWrapper;
import function.decorators.DerivativeFunctionWrapper;
import function.decorators.DimensionFocusWrapper;
import gradient.GradientDescent;
import gradient.updater.CompositeGradientUpdater;
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
import optimization.fittnesEvaluator.observable.Best1DFunctionSolutionGraphngObserver;
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

public class MultyModalFunctionParameterDetection {

	private MultyModalFunctionParameterDetection() {}
	
	
	public static void main(String[] args) {
		
		double learningRate = 0.1;
		double acceptableErrorRate = 10e-6;
		int maximumNumberOfGenerations = 1_000_000;
		
		//Function to optimize
		IFunction<double[]> multyModalFunction = new IFunction<double[]>() {
			
			@Override
			public int getVariableCount() {
				return 1;
			}
			
			@Override
			public double applyAsDouble(double[] X) {
				return Math.abs(Math.sin(0.8*X[0]))+0.5*Math.abs(X[0]);
			}
		};
		DerivativeFunctionCallCounterWrapper function = new DerivativeFunctionCallCounterWrapper(new DerivativeFunctionWrapper(multyModalFunction));		
		
		//Start UI
		SimpleGraph graph = new SimpleGraph(15,15);
		DoubleUnaryOperator function1D = new DimensionFocusWrapper(multyModalFunction, 0, new double[1]);
		graph.addFunction(function1D, Color.BLUE);
		graph.display();

		//Start solution
		IStartSolutionGenerator<DoubleArraySolution> startSolutionGenerator = new RandomStartSolutionGenerator(multyModalFunction.getVariableCount(), -10, 10);
		
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
				new PrintBestObserver<DoubleArraySolution,double[]>(System.out),
				new Best1DFunctionSolutionGraphngObserver<>(graph, function1D),
				//TODO: include to watch step by step progress
				new SleepOnBestObserver<DoubleArraySolution,double[]>(sleepTimeInMs)
		)));

		//Updater
		int variableCount = 1;
		double momentDecreaseFactor = 0.9;
		IGradientBasedUpdater updater = new CompositeGradientUpdater(Arrays.asList(
				new MomentGradientUpdater(learningRate, momentDecreaseFactor, variableCount )
				));
		
		//Optimization stopper
		IOptimisationStopper<DoubleArraySolution> stopper = new CompositeOptimisationStopper<>(Arrays.asList(
			new FunctionValueStopper<>(acceptableErrorRate),
			new GenerationNumberEvolutionStopper<>(maximumNumberOfGenerations)
		));
		
		//Optimization
		IOptimizationAlgorithm<DoubleArraySolution> optimizationAlgorithm = 
				new GradientDescent<DoubleArraySolution>(decoder, encoder , evaluator, updater, stopper, function, startSolutionGenerator);
				
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
