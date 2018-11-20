package gradient;

import javax.annotation.Nonnull;

import function.IGFunction;
import gradient.updater.IGradientBasedUpdater;
import optimization.algorithm.ISingleUnitOptimizationAlgorithm;
import optimization.decoder.IDecoder;
import optimization.encoder.IEncoder;
import optimization.fittnesEvaluator.IFitnessEvaluator;
import optimization.startSolutionGenerator.IStartSolutionGenerator;
import optimization.stopper.IOptimisationStopper;

public class GradientDescent<T> implements ISingleUnitOptimizationAlgorithm<T>{
	
	
	private final @Nonnull IDecoder<T,double[]> decoder;
	private final @Nonnull IEncoder<T> encoder;
	private final @Nonnull IFitnessEvaluator<T> fitnessEvaluator;
	private final @Nonnull IGradientBasedUpdater updater;
	private final @Nonnull IOptimisationStopper<T> stopper;
	//
	private final @Nonnull IGFunction gFunction;
	private IStartSolutionGenerator<T> startSolutionGenerator;

	public GradientDescent(IDecoder<T,double[]> decoder, IEncoder<T> encoder, 
			IFitnessEvaluator<T> fitnessEvaluator, 
			IGradientBasedUpdater updater, 
			IOptimisationStopper<T> stopper, 
			IGFunction gFunction, 
			IStartSolutionGenerator<T> startSolutionGenerator) {
		this.decoder = decoder;
		this.encoder = encoder;
		this.fitnessEvaluator = fitnessEvaluator;
		this.updater = updater;
		this.stopper = stopper;
		this.gFunction = gFunction;
		this.startSolutionGenerator = startSolutionGenerator;
	}
		
	public void resetOptimization(){
		stopper.reset();
	}
	
	@Override
	public @Nonnull T run() {
				
		T solution = startSolutionGenerator.generate();		
		
		double[] decodedSolution = decoder.decode(solution);
		double[] gradient = new double[gFunction.getVariableCount()];
		double value = gFunction.getValueAndGradient(decodedSolution, gradient);
		double fitness = fitnessEvaluator.evaluate(solution, value);
				
		while(!stopper.shouldStop(solution, fitness, value)){
			updater.update(decodedSolution, gradient, fitness, value);
			encoder.encode(decodedSolution, solution);
						
			value = gFunction.getValueAndGradient(decodedSolution, gradient);
			fitness = fitnessEvaluator.evaluate(solution, value);			
		}
				
		return solution;
	}
	
}
