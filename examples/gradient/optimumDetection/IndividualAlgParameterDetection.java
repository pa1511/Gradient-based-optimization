package gradient.optimumDetection;

import java.util.Arrays;

import function.common.benchmark.IBenchmarkFunction;
import function.common.benchmark.MatyasFunction;
import function.common.benchmark.RastriginFunction;
import function.common.benchmark.RosenbrockBananaFunction;
import function.common.benchmark.SchaffersFunction6;
import function.common.benchmark.SchaffersFunction7;
import function.common.benchmark.SphereFunction;
import function.common.benchmark.ThreeHumpCamelFunction;
import function.decorators.DerivativeFunctionCallCounterWrapper;
import function.decorators.DerivativeFunctionWrapper;
import gradient.IndividualOptimizer;
import gradient.individualityFactorProvider.ConstantIndividualityFactorProvider;
import gradient.individualityFactorProvider.IIndividualityFactorProvider;
import gradient.individualityFactorProvider.RandomIndividualityFactorProvider;
import gradient.updater.IGradientBasedUpdater;
import gradient.updater.SimpleGradientUpdater;
import optimization.algorithm.IOptimizationAlgorithm;
import optimization.algorithm.decorator.TimedOptimizationAlgorithm;
import optimization.decoder.PassThroughDoubleDecoder;
import optimization.fittnesEvaluator.FunctionValueFitnessEvaluator;
import optimization.fittnesEvaluator.ThroughOneFitnessEvaluator;
import optimization.fittnesEvaluator.observable.All2DUnitGraphingObservr;
import optimization.fittnesEvaluator.observable.Best2DUnitGraphngObserver;
import optimization.fittnesEvaluator.observable.BestObserver;
import optimization.fittnesEvaluator.observable.PerChromosomeObservableFitnessEvaluator;
import optimization.fittnesEvaluator.observable.PrintBestObserver;
import optimization.fittnesEvaluator.observable.SleepOnBestObserver;
import optimization.solution.DoubleArraySolution;
import optimization.startPopulationGenerator.IStartPopulationGenerator;
import optimization.startPopulationGenerator.RandomStartPopulationGenerator;
import optimization.stopper.CompositeOptimisationStopper;
import optimization.stopper.FunctionValueStopper;
import optimization.stopper.GenerationNumberEvolutionStopper;
import optimization.stopper.IOptimisationStopper;
import optimization.utility.AlgorithmsPresentationUtility;
import ui.graph.SimpleGraph;

public class IndividualAlgParameterDetection {

	public static void main(String[] args) {

		//Parameters
		double individualityFactor = 0.6;
		int populationSize = 5;
		double learningRate = 0.05;
		double acceptableErrorRate = 1e-3;
		int maximumNumberOfGenerations = 50_000;

		
		// Function
		IBenchmarkFunction benchFunction = //new RastriginFunction();
									//new MatyasFunction();
									//new RosenbrockBananaFunction();
									//new SchaffersFunction6(2);
									//new SchaffersFunction7(2);
									new SphereFunction();
									//new ThreeHumpCamelFunction();
				
		DerivativeFunctionCallCounterWrapper function = new DerivativeFunctionCallCounterWrapper(
				new DerivativeFunctionWrapper(benchFunction));
		int variableCount = function.getVariableCount();

		// Start population generator
		IStartPopulationGenerator<DoubleArraySolution> startPopulationGenerator = new RandomStartPopulationGenerator(
				populationSize, variableCount, 
				-50,-25
				//benchFunction.getStandardSearchMin(), benchFunction.getStandardSearchMax()
				);

		//IndividualityFactor provider
		IIndividualityFactorProvider individualityFactorProvider = //new ConstantIndividualityFactorProvider(individualityFactor);
											new RandomIndividualityFactorProvider();
						
		// Gradient update
		IGradientBasedUpdater updater = new SimpleGradientUpdater(learningRate, variableCount);
							
		//Decoder
		PassThroughDoubleDecoder decoder = new PassThroughDoubleDecoder();

		
		//Start UI
		SimpleGraph graph = new SimpleGraph(12,12);
		graph.display();

		// Fitness evaluator
		// IFitnessEvaluator<DoubleArraySolution> evaluator = new ThroughOneFitnessEvaluator<>(new FunctionValueFitnessEvaluator<>());
		PerChromosomeObservableFitnessEvaluator<DoubleArraySolution> evaluator = new PerChromosomeObservableFitnessEvaluator<>(
				v -> ThroughOneFitnessEvaluator.evaluationMethod
						.applyAsDouble(FunctionValueFitnessEvaluator.evaluationMethod.applyAsDouble(v)));
		int sleepTimeInMs = 50;
		evaluator.addObserver(new BestObserver<>(decoder,
				Arrays.asList(new PrintBestObserver<DoubleArraySolution, double[]>(System.out),
//						new Best2DUnitGraphngObserver<>(graph),
						// TODO: include to watch step by step progress
						new SleepOnBestObserver<DoubleArraySolution, double[]>(sleepTimeInMs))));
		evaluator.addObserver(new All2DUnitGraphingObservr<DoubleArraySolution>(graph, decoder));

		// Optimization stopper
		IOptimisationStopper<DoubleArraySolution> stopper = new CompositeOptimisationStopper<>(
				Arrays.asList(new FunctionValueStopper<>(acceptableErrorRate),
						new GenerationNumberEvolutionStopper<>(maximumNumberOfGenerations)));

		// Optimization algorithm
		IOptimizationAlgorithm<DoubleArraySolution> optimizationAlgorithm = new IndividualOptimizer(function,
				startPopulationGenerator, individualityFactorProvider, updater, evaluator, stopper);
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
