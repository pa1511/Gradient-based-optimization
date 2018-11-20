package gradient.updater;

import java.util.Arrays;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

import util.VectorUtils;

public class MomentGradientUpdater implements IGradientBasedUpdater{

	private final double momentDecreaseFactor;
	private final @Nonnull double[] learningRates;
	private final @Nonnull double[] gradientUpdateField;

	public MomentGradientUpdater(@Nonnegative double learningRate,@Nonnegative double momentDecreaseFactor,@Nonnegative int variableCount) {
		this(new double[variableCount],momentDecreaseFactor,variableCount);
		Arrays.fill(this.learningRates, learningRate);
	}

	public MomentGradientUpdater(@Nonnull double[] learningRates,@Nonnegative double momentDecreaseFactor,@Nonnegative int variableCount) {
		this.learningRates = learningRates;
		this.momentDecreaseFactor = momentDecreaseFactor;
		this.gradientUpdateField = new double[variableCount];
	}

	
	@Override
	public void update(@Nonnull double[] values,@Nonnull double[] gradients, double fintess, double value) {
		VectorUtils.multiplyByScalar(gradientUpdateField, momentDecreaseFactor, gradientUpdateField, false);
		VectorUtils.add(gradientUpdateField, gradients, gradientUpdateField, false);
		for(int i=0; i<values.length;i++){
			values[i]-=gradientUpdateField[i]*learningRates[i];
		}
	}

}
