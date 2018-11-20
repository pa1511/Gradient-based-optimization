package gradient.optimumDetection;

import java.util.Arrays;

import function.common.benchmark.IBenchmarkFunction;
import function.common.benchmark.AckleyFunction;
import function.common.benchmark.EggholderFunction;
import function.common.benchmark.HimmelblausFunction;
import function.common.benchmark.MatyasFunction;
import function.common.benchmark.RastriginFunction;
import function.common.benchmark.SphereFunction;
import function.decorators.DerivativeFunctionCallCounterWrapper;
import function.decorators.DerivativeFunctionWrapper;
import gradient.PsychoticOptimizer;
import gradient.updater.IGradientBasedUpdater;
import gradient.updater.RestrictedGradientUpdater;
import gradient.updater.SimpleGradientUpdater;
import optimization.algorithm.IOptimizationAlgorithm;
import optimization.algorithm.decorator.TimedOptimizationAlgorithm;
import optimization.decoder.PassThroughDoubleDecoder;
import optimization.fittnesEvaluator.FunctionValueFitnessEvaluator;
import optimization.fittnesEvaluator.NegateFitnessEvaluator;
import optimization.fittnesEvaluator.observable.BestObserver;
import optimization.fittnesEvaluator.observable.PerChromosomeObservableFitnessEvaluator;
import optimization.fittnesEvaluator.observable.PrintBestObserver;
import optimization.solution.DoubleArraySolution;
import optimization.startPopulationGenerator.IStartPopulationGenerator;
import optimization.startPopulationGenerator.RandomStartPopulationGenerator;
import optimization.stopper.CompositeOptimisationStopper;
import optimization.stopper.FunctionEvaluationCountStopper;
import optimization.stopper.FunctionValueStopper;
import optimization.stopper.IOptimisationStopper;
import optimization.utility.AlgorithmsPresentationUtility;

public class PsyhoticAlgParameterDetection {


	public static void main(String[] args) {

		//Parameters
		double beleiveBest = 0.8;
		int populationSize = 50;
		double learningRate = 0.002;
		double acceptableErrorRate = 1e-2;
		int maxFunctionEvaluations = 5_000_000;

		
		// Function
		IBenchmarkFunction benchFunction = 
									//new RastriginFunction();
									//new SphereFunction();
									//new MatyasFunction();
									new HimmelblausFunction();
									//new EggholderFunction();
									//new AckleyFunction();
				
		DerivativeFunctionCallCounterWrapper function = new DerivativeFunctionCallCounterWrapper(
				new DerivativeFunctionWrapper(benchFunction));
		int variableCount = function.getVariableCount();

		// Start population generator
		IStartPopulationGenerator<DoubleArraySolution> startPopulationGenerator = new RandomStartPopulationGenerator(
				populationSize, variableCount, 
				benchFunction.getStandardSearchMin(),benchFunction.getStandardSearchMax()
				);						
		// Gradient update
		IGradientBasedUpdater updater = new RestrictedGradientUpdater(new SimpleGradientUpdater(learningRate, variableCount),
				benchFunction.getStandardSearchMin(),benchFunction.getStandardSearchMax());
							
		//Decoder
		PassThroughDoubleDecoder decoder = new PassThroughDoubleDecoder();

		
		// Fitness evaluator
		
		PerChromosomeObservableFitnessEvaluator<DoubleArraySolution> evaluator = new PerChromosomeObservableFitnessEvaluator<>(
				v -> NegateFitnessEvaluator.evaluationMethod
						.applyAsDouble(FunctionValueFitnessEvaluator.evaluationMethod.applyAsDouble(v)));
		evaluator.addObserver(new BestObserver<>(decoder,
				Arrays.asList(new PrintBestObserver<DoubleArraySolution, double[]>(System.out)
				)));

		// Optimization stopper
		IOptimisationStopper<DoubleArraySolution> stopper = new CompositeOptimisationStopper<>(Arrays.asList(
						new FunctionValueStopper<>(benchFunction.getMinValue(),acceptableErrorRate),
						new FunctionEvaluationCountStopper<>(function, maxFunctionEvaluations)
						));

		// Optimization algorithm
		IOptimizationAlgorithm<DoubleArraySolution> optimizationAlgorithm = new PsychoticOptimizer(function,
				startPopulationGenerator, evaluator, updater, beleiveBest, stopper);
		TimedOptimizationAlgorithm<DoubleArraySolution> timedOptAlgorithm = new TimedOptimizationAlgorithm<>(
				optimizationAlgorithm);

		// Solution presentation
		DoubleArraySolution solution = timedOptAlgorithm.run();
		System.out.println();
		AlgorithmsPresentationUtility.printExecutionTime(timedOptAlgorithm.getExecutionTime());
		System.out.println("Solution: " + solution);
		System.out.println("Optimum solution: " + Arrays.toString(benchFunction.getMinValueCoordinates()));
		System.out.println("Value: " + function.applyAsDouble(solution.values) + "\n");
		AlgorithmsPresentationUtility.printEvaluationCount(function.getEvaluationCount());
	}

}
