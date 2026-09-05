#ifndef NOTORIOUS6_ICTCP_GLSL
#define NOTORIOUS6_ICTCP_GLSL

// From https://www.shadertoy.com/view/ldKcz3

const float PQ_C1 = 0.8359375f;      // 3424.f / 4096.f;
const float PQ_C2 = 18.8515625f;     // 2413.f / 4096.f * 32.f;
const float PQ_C3 = 18.6875f;        // 2392.f / 4096.f * 32.f;
const float PQ_M1 = 0.159301758125f; // 2610.f / 4096.f / 4;
const float PQ_M2 = 78.84375f;       // 2523.f / 4096.f * 128.f;
const float PQ_MAX = 10000.0;

// PQ_OETF - Optical-Electro Transfer Function

float linear_to_PQ(float linearValue)
{
	float L = linearValue / PQ_MAX;
	float Lm1 = pow(L, PQ_M1);
	float X = (PQ_C1 + PQ_C2 * Lm1) / (1.0f + PQ_C3 * Lm1);
	float pqValue = pow(X, PQ_M2);
	return pqValue;
}

vec3 linear_to_PQ(vec3 linearValues)
{
	vec3 L = linearValues / PQ_MAX;
	vec3 Lm1 = pow(max(0.0.xxx, L.xyz), PQ_M1.xxx);
	vec3 X = (PQ_C1 + PQ_C2 * Lm1) / (1.0f + PQ_C3 * Lm1);
	vec3 pqValues = pow(max(0.0.xxx, X), PQ_M2.xxx);
	return pqValues;
}

// PQ_EOTF - Electro-Optical Transfer Function

float PQ_to_linear(float pqValue)
{
	float M = PQ_C2 - PQ_C3 * pow(max(0.0, pqValue), 1.0 / PQ_M2);
	float N = max(pow(max(0.0, pqValue), 1.0f / PQ_M2) - PQ_C1, 0.0f);
	float L = pow(N / M, 1.0f / PQ_M1);
	float linearValue = L * PQ_MAX;
	return linearValue;
}

vec3 PQ_to_linear(vec3 pqValues)
{
	vec3 M = PQ_C2 - PQ_C3 * pow(max(0.0.xxx, pqValues), 1. / PQ_M2.xxx);
	vec3 N = max(pow(max(0.0.xxx, pqValues), 1. / PQ_M2.xxx) - PQ_C1, 0.0f);
	vec3 L = pow(max(0.0.xxx, N / M), 1. / PQ_M1.xxx);
	vec3 linearValues = L * PQ_MAX;
	return linearValues;
}


// BT.709 <-> BT.2020 Primaries

vec3 BT709_to_BT2020(vec3 linearBT709)
{
	const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
		vec3(0.6274, 0.0691, 0.0164),
		vec3(0.3293, 0.9195, 0.0880),
		vec3(0.0433, 0.0113, 0.8956));
	return LINEAR_SRGB_TO_LINEAR_REC2020 * linearBT709.rgb;
}

vec3 BT2020_to_BT709(vec3 linearBT2020)
{
	const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
		vec3(1.6605, -0.1246, -0.0182),
		vec3(-0.5876, 1.1329, -0.1006),
		vec3(-0.0728, -0.0083, 1.1187));
	return LINEAR_REC2020_TO_LINEAR_SRGB * linearBT2020.rgb;
}

// LMS <-> BT2020

vec3 BT2020_to_LMS(vec3 linearBT2020)
{
	float R = linearBT2020.r;
	float G = linearBT2020.g;
	float B = linearBT2020.b;

	float L = 0.4121093750000000f * R + 0.5239257812500000f * G + 0.0639648437500000f * B;
	float M = 0.1667480468750000f * R + 0.7204589843750000f * G + 0.1127929687500000f * B;
	float S = 0.0241699218750000f * R + 0.0754394531250000f * G + 0.9003906250000000f * B;

	vec3 linearLMS = vec3(L, M, S);
	return linearLMS;
}

vec3 LMS_to_BT2020(vec3 linearLMS)
{
	float L = linearLMS.x;
	float M = linearLMS.y;
	float S = linearLMS.z;

	float R = 3.4366066943330793f * L - 2.5064521186562705f * M + 0.0698454243231915f * S;
	float G = -0.7913295555989289f * L + 1.9836004517922909f * M - 0.1922708961933620f * S;
	float B = -0.0259498996905927f * L - 0.0989137147117265f * M + 1.1248636144023192f * S;

	vec3 linearBT2020 = vec3(R, G, B);
	return linearBT2020;
}

// Misc. Color Space Conversion

// ICtCp <-> PQ LMS

vec3 PQ_LMS_to_ICtCp(vec3 PQ_LMS)
{
	float L = PQ_LMS.x;
	float M = PQ_LMS.y;
	float S = PQ_LMS.z;

	float I = 0.5f * L + 0.5f * M;
	float Ct = 1.613769531250000f * L - 3.323486328125000f * M + 1.709716796875000f * S;
	float Cp = 4.378173828125000f * L - 4.245605468750000f * M - 0.132568359375000f * S;

	vec3 ICtCp = vec3(I, Ct, Cp);
	return ICtCp;
}

vec3 ICtCp_to_PQ_LMS(vec3 ICtCp)
{
	float I = ICtCp.x;
	float Ct = ICtCp.y;
	float Cp = ICtCp.z;

	float L = I + 0.00860903703793281f * Ct + 0.11102962500302593f * Cp;
	float M = I - 0.00860903703793281f * Ct - 0.11102962500302593f * Cp;
	float S = I + 0.56003133571067909f * Ct - 0.32062717498731880f * Cp;

	vec3 PQ_LMS = vec3(L, M, S);
	return PQ_LMS;
}

// Linear BT2020 <-> ICtCp
//
// https://www.dolby.com/us/en/technologies/dolby-vision/ictcp-white-paper.pdf
// http://www.jonolick.com/home/hdr-videos-part-2-colors

vec3 BT2020_to_ICtCp(vec3 linearBT2020)
{
	vec3 LMS = BT2020_to_LMS(linearBT2020);
	vec3 PQ_LMS = linear_to_PQ(LMS);
	vec3 ICtCp = PQ_LMS_to_ICtCp(PQ_LMS);

	return ICtCp;
}

vec3 ICtCp_to_BT2020(vec3 ICtCp)
{
	vec3 PQ_LMS = ICtCp_to_PQ_LMS(ICtCp);
	vec3 LMS = PQ_to_linear(PQ_LMS);
	vec3 linearBT2020 = LMS_to_BT2020(LMS);
	return linearBT2020;
}

// ----------------------------------------------------------------

vec3 BT709_to_ICtCp(vec3 linearBT709)
{
	vec3 linearBT2020 = BT709_to_BT2020(linearBT709);
	vec3 LMS = BT2020_to_LMS(linearBT2020);
	vec3 PQ_LMS = linear_to_PQ(LMS);
	vec3 ICtCp = PQ_LMS_to_ICtCp(PQ_LMS);

	return ICtCp;
}

vec3 ICtCp_to_BT709(vec3 ICtCp)
{
	vec3 PQ_LMS = ICtCp_to_PQ_LMS(ICtCp);
	vec3 LMS = PQ_to_linear(PQ_LMS);
	vec3 linearBT2020 = LMS_to_BT2020(LMS);
	vec3 linearBT709 = BT2020_to_BT709(linearBT2020);
	return linearBT709;
}

#endif
