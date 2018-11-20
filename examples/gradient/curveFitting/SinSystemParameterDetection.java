package gradient.curveFitting;

import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import dataset.handeling.DataSetLoader;
import function.decorators.DerivativeFunctionCallCounterWrapper;
import function.decorators.DimensionFocusWrapper;
import function.error.PrototypeBasedSystemLossFunction;
import function.error.SquareErrorFunction;
import function.prototype.SinFunction;
import gradient.GradientDescent;
import gradient.updater.IGradientBasedUpdater;
import gradient.updater.SimpleGradientUpdater;
import optimization.algorithm.IOptimizationAlgorithm;
import optimization.algorithm.decorator.TimedOptimizationAlgorithm;
import optimization.decoder.IDecoder;
import optimization.decoder.PassThroughDoubleDecoder;
import optimization.encoder.IEncoder;
import optimization.encoder.PassThroughEncoder;
import optimization.fittnesEvaluator.FunctionValueFitnessEvaluator;
import optimization.fittnesEvaluator.ThroughOneFitnessEvaluator;
import optimization.fittnesEvaluator.observable.BestObserver;
import optimization.fittnesEvaluator.observable.PerChromosomeObservableFitnessEvaluator;
import optimization.fittnesEvaluator.observable.PrintBestObserver;
import optimization.fittnesEvaluator.observable.PrototypeGraphngBestObserver;
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

public class SinSystemParameterDetection {

	private SinSystemParameterDetection() {}
	
	
	public static void main(String[] args) throws IOException {
		
		double acceptableErrorRate = 10e-8;
		int maximumNumberOfGenerations = 1_000_000;
		
		//Function to optimize
		SinFunction sinFunction = new SinFunction();
		double[][] systemMatrix = DataSetLoader.loadMatrix(new File(System.getProperty("user.dir"),"data/sin-data.txt"));
		DerivativeFunctionCallCounterWrapper function =  new DerivativeFunctionCallCounterWrapper(
				new PrototypeBasedSystemLossFunction(systemMatrix,sinFunction,new SquareErrorFunction()));		
		
		//Start UI
		SimpleGraph graph = new SimpleGraph(10,4);
		graph.addFunction(new DimensionFocusWrapper(sinFunction, 0, new double[1]), Color.BLUE);
		for(double[] row:systemMatrix){
			graph.addPoint(row[0], row[1]);
		}
		graph.display();
		
		//Start solution
		int variableNumber = sinFunction.getCoefficientCount();
		IStartSolutionGenerator<DoubleArraySolution> startSolutionGenerator = 
				new RandomStartSolutionGenerator(variableNumber);
		
		//Decoder
		IDecoder<DoubleArraySolution,double[]> decoder = new PassThroughDoubleDecoder();

		//Encoder
		IEncoder<DoubleArraySolution> encoder = new PassThroughEncoder();

		//Fitness evaluator
		PerChromosomeObservableFitnessEvaluator<DoubleArraySolution> evaluator = new PerChromosomeObservableFitnessEvaluator<>(v->
			 ThroughOneFitnessEvaluator.evaluationMethod.applyAsDouble(
						FunctionValueFitnessEvaluator.evaluationMethod.applyAsDouble(v)
		));
		int sleepTimeInMs = 250;
		evaluator.addObserver(new BestObserver<>(decoder, Arrays.asList(
				new PrintBestObserver<DoubleArraySolution,double[]>(System.out),
				new PrototypeGraphngBestObserver<DoubleArraySolution>(sinFunction, graph),
				//TODO: include to watch step by step progress
				new SleepOnBestObserver<DoubleArraySolution,double[]>(sleepTimeInMs)
		)));
						
		//Updater
		double learningRate = 0.25;
		IGradientBasedUpdater updater = new SimpleGradientUpdater(learningRate,variableNumber);
		
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
		System.out.println("Error: " + function.applyAsDouble(solution.getValues()));
		AlgorithmsPresentationUtility.printEvaluationCount(function.getEvaluationCount());
	}
	
}
