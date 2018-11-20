package gradient.curveFitting;

import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import dataset.handeling.DataSetLoader;
import function.IGFunction;
import function.decorators.DerivativeFunctionCallCounterWrapper;
import function.decorators.DimensionFocusWrapper;
import function.decorators.FunctionScaleWrapper;
import function.error.PrototypeBasedSystemLossFunction;
import function.error.SquareErrorFunction;
import function.prototype.APrototypeFunction;
import function.prototype.ExpFunction;
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

public class ExpSystemParameterDetection {

	private ExpSystemParameterDetection() {}
	
	
	public static void main(String[] args) throws IOException {
		
		double acceptableErrorRate = 10e-6;
		int maximumNumberOfGenerations = 1_000_000;
		
		//Function to optimize
		APrototypeFunction expFunction = new ExpFunction();
		int variableCount = expFunction.getCoefficientCount();
		double[][] systemMatrix = DataSetLoader.loadMatrix(new File(System.getProperty("user.dir"),"data/exp-data.txt"));
		IGFunction prototypeBasedSystemLossFunction = new FunctionScaleWrapper(
				new PrototypeBasedSystemLossFunction(systemMatrix,expFunction,new SquareErrorFunction(),1+systemMatrix.length/5),
				1.0/systemMatrix.length);
		DerivativeFunctionCallCounterWrapper function =  new DerivativeFunctionCallCounterWrapper(prototypeBasedSystemLossFunction);		
		
		//Start UI
		SimpleGraph graph = new SimpleGraph(4,4);
		graph.addFunction(new DimensionFocusWrapper(expFunction, 0, new double[1]), Color.BLUE);
		for(double[] row:systemMatrix){
			graph.addPoint(row[0], row[1]);
		}
		graph.display();
		
		try {
			Thread.sleep(8000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//Start solution
		int variableNumber = expFunction.getCoefficientCount();
		IStartSolutionGenerator<DoubleArraySolution> startSolutionGenerator = new RandomStartSolutionGenerator(variableNumber);
		
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
				new PrototypeGraphngBestObserver<DoubleArraySolution>(expFunction, graph)
				,
				//TODO: include to watch step by step progress
				new SleepOnBestObserver<DoubleArraySolution,double[]>(sleepTimeInMs)
		)));
						
		//Updater
		double learningRate = 0.005;
		double momentDecreaseFactor = 0.8;
		IGradientBasedUpdater updater = new MomentGradientUpdater(learningRate, momentDecreaseFactor , variableCount);
		
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
