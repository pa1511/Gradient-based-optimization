package gradient;

import java.util.Random;

import javax.annotation.Nonnull;

import function.IGFunction;
import gradient.updater.IGradientBasedUpdater;
import optimization.algorithm.IOptimizationAlgorithm;
import optimization.fittnesEvaluator.IFitnessEvaluator;
import optimization.solution.DoubleArraySolution;
import optimization.startPopulationGenerator.IStartPopulationGenerator;
import optimization.stopper.IOptimisationStopper;
import optimization.utility.OptimizationAlgorithmsUtility;
import utilities.random.RNGProvider;

public class PsychoticOptimizer implements IOptimizationAlgorithm<DoubleArraySolution>{

	private final @Nonnull IGFunction function;
	private final @Nonnull IStartPopulationGenerator<DoubleArraySolution> startPopulationGenerator;
	private final @Nonnull IFitnessEvaluator<DoubleArraySolution> evaluator;
	private final @Nonnull IGradientBasedUpdater gradientUpdater;
	private double believeBest;
	private final @Nonnull IOptimisationStopper<DoubleArraySolution> stopper;


	public PsychoticOptimizer(@Nonnull IGFunction function,
			@Nonnull IStartPopulationGenerator<DoubleArraySolution> startPopulationGenerator,
			@Nonnull IFitnessEvaluator<DoubleArraySolution> evaluator,
			@Nonnull IGradientBasedUpdater gradientUpdater,
			double believeBest,
			@Nonnull IOptimisationStopper<DoubleArraySolution> stopper) {
				this.function = function;
				this.startPopulationGenerator = startPopulationGenerator;
				this.evaluator = evaluator;
				this.gradientUpdater = gradientUpdater;
				this.believeBest = believeBest;
				this.stopper = stopper;
	}
	
	
	@Override
	public DoubleArraySolution run() {
		
		Random random = RNGProvider.getRandom();
		
		DoubleArraySolution[] solution = startPopulationGenerator.generate();
		double[] solutionFunctionValue = new double[solution.length];
		double[] solutionQuality = new double[solution.length];
		
		for(int i=0; i<solution.length; i++) {
			solutionFunctionValue[i] = function.applyAsDouble(solution[i].values);
			solutionQuality[i] = evaluator.evaluate(solution[i], solutionFunctionValue[i]);
		}
		
		double[] gradients = new double[solution[0].values.length];
		
		while(!stopper.shouldStop(solution, solutionQuality, solutionFunctionValue)) {
			int best = OptimizationAlgorithmsUtility.getBestSolutionIndex(solution, solutionQuality);

			for(int i=0; i<solution.length; i++) {
				if(i==best) {
					//calculate gradients
					function.getGradientValueAt(solution[i].values,gradients);
					//update solution
					gradientUpdater.update(solution[i].values, gradients, solutionQuality[i], solutionFunctionValue[i]);
					//update function value and quality
					solutionFunctionValue[i] = function.applyAsDouble(solution[i].values);
					solutionQuality[i] = evaluator.evaluate(solution[i], solutionFunctionValue[i]);
				}
				else {
					if(random.nextDouble()<=believeBest) {//TODO: this could be determined in a better way
						for(int j=0; j<gradients.length; j++) {
							gradients[j] = (random.nextDouble()-0.5)*2;
						}
					}
					else {
						//calculate gradients
						function.getGradientValueAt(solution[i].values,gradients);
					}
					//update solution
					gradientUpdater.update(solution[i].values, gradients, solutionQuality[i], solutionFunctionValue[i]);
					//update function value and quality
					solutionFunctionValue[i] = function.applyAsDouble(solution[i].values);
					solutionQuality[i] = evaluator.evaluate(solution[i], solutionFunctionValue[i]);
				}
			}
			
		}
		
		return OptimizationAlgorithmsUtility.getBestSolution(solution, solutionQuality);
	}

}
