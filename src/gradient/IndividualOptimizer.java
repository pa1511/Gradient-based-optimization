package gradient;

import function.IGFunction;
import gradient.individualityFactorProvider.IIndividualityFactorProvider;
import gradient.updater.IGradientBasedUpdater;
import optimization.algorithm.IOptimizationAlgorithm;
import optimization.fittnesEvaluator.IFitnessEvaluator;
import optimization.solution.DoubleArraySolution;
import optimization.startPopulationGenerator.IStartPopulationGenerator;
import optimization.stopper.IOptimisationStopper;
import optimization.utility.OptimizationAlgorithmsUtility;

public class IndividualOptimizer implements IOptimizationAlgorithm<DoubleArraySolution>{

	private final IGFunction function;
	private final IStartPopulationGenerator<DoubleArraySolution> startPopulationGenerator;
	private final IIndividualityFactorProvider individualityFactorProvider;
	private final IGradientBasedUpdater gradientBasedUpdater;
	private final IFitnessEvaluator<DoubleArraySolution> evaluator;
	private final IOptimisationStopper<DoubleArraySolution> optimisationStopper;


	public IndividualOptimizer(IGFunction function, IStartPopulationGenerator<DoubleArraySolution> startPopulationGenerator, IIndividualityFactorProvider individualityFactorProvider,IGradientBasedUpdater gradientBasedUpdater,IFitnessEvaluator<DoubleArraySolution> evaluator, IOptimisationStopper<DoubleArraySolution> optimisationStopper) {
		this.function = function;
		this.startPopulationGenerator = startPopulationGenerator;
		this.individualityFactorProvider = individualityFactorProvider;
		this.gradientBasedUpdater = gradientBasedUpdater;
		this.evaluator = evaluator;
		this.optimisationStopper = optimisationStopper;
	}
	
	
	@Override
	public DoubleArraySolution run() {
		
		DoubleArraySolution[] population = startPopulationGenerator.generate();
		double[] populationFunctionValues = new double[population.length];
		double[] populationFitnessValues = new double[population.length];

		//replace the first with centroid !
		population[0] = DoubleArraySolution.calculateCentroid(population, 0); 
					
		//evaluate the whole population 
		for(int i=0; i<population.length; i++) {
			populationFunctionValues[i] = function.applyAsDouble(population[i].values);
			populationFitnessValues[i] = evaluator.evaluate(population[i], populationFunctionValues[i]);
		}

		
		
		do {
			
//			//place best solution at last place
//			int bestId = OptimizationAlgorithmsUtility.getBestSolutionIndex(population, populationFitnessValues);
//			if(bestId!=0){
//				if(bestId!=population.length-1) {
//					DoubleArraySolution swap = population[bestId];
//					double swapFunctionValue = populationFunctionValues[bestId];
//					double swapFitnessValue = populationFitnessValues[bestId];
//					
//					population[bestId] = population[population.length-1];
//					populationFunctionValues[bestId] = populationFunctionValues[population.length-1];
//					populationFitnessValues[bestId] = populationFitnessValues[population.length-1];
//					
//					population[population.length-1] = swap;
//					populationFunctionValues[population.length-1] = swapFunctionValue;
//					populationFitnessValues[population.length-1] = swapFitnessValue;
//				}
//			}
//			else  {
//				population[population.length-1].copy(population[0]);
//				populationFunctionValues[population.length-1] = populationFunctionValues[0];
//				populationFitnessValues[population.length-1] = populationFitnessValues[0];
//			}
			
			//calculating new points except for centroid
			double[] centroidGradient = function.getGradientValueAt(population[0].values);
			double[] solutionGradient = new double[centroidGradient.length];
			double[] gradient = new double[solutionGradient.length];
			
			
			for(int i=1; i<population.length/*-1*/;i++) {//TODO: -1 when placing best at last
				DoubleArraySolution solution = population[i];
				function.getGradientValueAt(solution.values, solutionGradient);
				double individualityFactor = individualityFactorProvider.getIndividualityFactor();
				
				for(int j=0; j<gradient.length; j++) {
					gradient[j] = solutionGradient[j]*individualityFactor + centroidGradient[j]*(1-individualityFactor);
				}

				gradientBasedUpdater.update(solution.values, gradient, populationFitnessValues[i], populationFunctionValues[i]);
				
				populationFunctionValues[i] = function.applyAsDouble(solution.values);
				populationFitnessValues[i] = evaluator.evaluate(solution, populationFunctionValues[i]);
			}
			
			
			//replace the first with centroid !
			population[0] = DoubleArraySolution.calculateCentroid(population, 0); 
			populationFunctionValues[0] = function.applyAsDouble(population[0].values);
			populationFitnessValues[0] = evaluator.evaluate(population[0], populationFunctionValues[0]);
			
			
		}while(!optimisationStopper.shouldStop(population, populationFitnessValues, populationFunctionValues));

				
		return OptimizationAlgorithmsUtility.getBestSolution(population, populationFitnessValues);
	}

}
