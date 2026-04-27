using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Face_Recognition.Services
{
    public class NidData
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("name_en")]
        public string NameEn { get; set; } = string.Empty;

        [JsonPropertyName("name_bn")]
        public string NameBn { get; set; } = string.Empty;

        [JsonPropertyName("father_bn")]
        public string FatherBn { get; set; } = string.Empty;

        [JsonPropertyName("mother_bn")]
        public string MotherBn { get; set; } = string.Empty;

        [JsonPropertyName("nid_number")]
        public string NidNumber { get; set; } = string.Empty;

        [JsonPropertyName("dob")]
        public string Dob { get; set; } = string.Empty;
    }

    public class IdValidationResult
    {
        public bool   Valid   { get; set; }
        public string IdType  { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public string? Error  { get; set; }
    }

    public class FaceVerificationResult
    {
        [JsonPropertyName("match")]
        public bool Match { get; set; }

        [JsonPropertyName("message")]
        public string Message { get; set; } = string.Empty;

        [JsonPropertyName("similarity")]
        public double Similarity { get; set; }

        public double MatchPercentage => Math.Round(Math.Max(0, Math.Min(100, Similarity * 100)), 1);

        public string MatchStatus
        {
            get
            {
                if (Error != null) return "Error";
                if (ConsistencyOk == false) return "Spoofing Detected";
                if (Similarity >= 0.55) return "Verified";
                if (Similarity >= 0.35) return "Needs Review";
                return "Not Verified";
            }
        }

        [JsonPropertyName("confidence")]
        public string Confidence { get; set; } = string.Empty;

        [JsonPropertyName("threshold")]
        public double Threshold { get; set; }

        [JsonPropertyName("method")]
        public string Method { get; set; } = string.Empty;

        [JsonPropertyName("nid_data")]
        public NidData? NidData { get; set; }

        [JsonPropertyName("nid_face_image")]
        public string? NidFaceImage { get; set; }

        [JsonPropertyName("live_face_image")]
        public string? LiveFaceImage { get; set; }

        public bool? NameMatch   { get; set; }
        public bool? NameBnMatch { get; set; }
        public bool? FatherMatch { get; set; }
        public bool? MotherMatch { get; set; }
        public bool? NidMatch    { get; set; }
        public bool? DobMatch    { get; set; }

        public string? UserName   { get; set; }
        public string? UserNameBn { get; set; }
        public string? UserFather { get; set; }
        public string? UserMother { get; set; }
        public string? UserNid    { get; set; }
        public string? UserDob    { get; set; }

        public string? Error { get; set; }

        [JsonPropertyName("consistency_ok")]
        public bool? ConsistencyOk { get; set; }

        [JsonPropertyName("consistency_scores")]
        public Dictionary<string, double>? ConsistencyScores { get; set; }

        [JsonPropertyName("liveness_captures")]
        public int LivenessCaptures { get; set; } = 3;

        public bool SpoofFlagged => ConsistencyOk == false;
    }

    public class FaceVerificationService
    {
        private readonly HttpClient _httpClient;

        public FaceVerificationService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            PropertyNameCaseInsensitive = true
        };

        public async Task<IdValidationResult> ValidateIdAsync(IFormFile idImage)
        {
            try
            {
                using var idStream = new MemoryStream();
                await idImage.CopyToAsync(idStream);

                using var form      = new MultipartFormDataContent();
                var idContent       = new ByteArrayContent(idStream.ToArray());
                idContent.Headers.ContentType = new MediaTypeHeaderValue(idImage.ContentType ?? "image/jpeg");
                form.Add(idContent, "id_image", idImage.FileName ?? "id_image.jpg");

                var response = await _httpClient.PostAsync("/validate-id", form);
                var json     = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    using var errDoc = JsonDocument.Parse(json);
                    var detail = errDoc.RootElement.TryGetProperty("detail", out var d)
                        ? d.GetString() : "ID validation failed.";
                    return new IdValidationResult { Valid = false, Error = detail };
                }

                return JsonSerializer.Deserialize<IdValidationResult>(json, _jsonOptions)
                    ?? new IdValidationResult { Valid = false, Error = "Empty response from API." };
            }
            catch (HttpRequestException ex)
            {
                return new IdValidationResult
                {
                    Valid = false,
                    Error = $"Cannot connect to Python API. Ensure Docker is running. ({ex.Message})"
                };
            }
            catch (Exception ex)
            {
                return new IdValidationResult { Valid = false, Error = $"Unexpected error: {ex.Message}" };
            }
        }

        public async Task<FaceVerificationResult> VerifyWithLivenessAsync(
            IFormFile idImage,
            string    sessionId)
        {
            try
            {
                using var idStream = new MemoryStream();
                await idImage.CopyToAsync(idStream);

                using var form = new MultipartFormDataContent();
                var idContent  = new ByteArrayContent(idStream.ToArray());
                idContent.Headers.ContentType = new MediaTypeHeaderValue(idImage.ContentType ?? "image/jpeg");
                form.Add(idContent, "id_image", idImage.FileName ?? "id_image.jpg");

                // session_id is a query param — Python reads the selfie from the session
                var url      = $"/verify-with-liveness?session_id={Uri.EscapeDataString(sessionId)}";
                var response = await _httpClient.PostAsync(url, form);
                var json     = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    string detail;
                    bool isSpoof = (int)response.StatusCode == 403;
                    try
                    {
                        using var errDoc = JsonDocument.Parse(json);
                        detail = errDoc.RootElement.TryGetProperty("detail", out var d)
                            ? d.GetString() ?? "Face verification failed."
                            : "Face verification failed.";
                    }
                    catch
                    {
                        detail = "Face verification failed.";
                    }

                    if (isSpoof)
                    {
                        return new FaceVerificationResult
                        {
                            Error         = null,
                            Match         = false,
                            ConsistencyOk = false,
                            Message       = detail
                        };
                    }

                    return new FaceVerificationResult { Error = detail };
                }

                return JsonSerializer.Deserialize<FaceVerificationResult>(json, _jsonOptions)
                    ?? new FaceVerificationResult { Error = "Empty response from verification API." };
            }
            catch (TaskCanceledException)
            {
                return new FaceVerificationResult
                {
                    Error = "Verification request timed out. The ArcFace model may still be loading — please try again in 30 seconds."
                };
            }
            catch (HttpRequestException ex)
            {
                return new FaceVerificationResult
                {
                    Error = $"Cannot reach Python API. Ensure Docker containers are running. ({ex.Message})"
                };
            }
            catch (Exception ex)
            {
                return new FaceVerificationResult { Error = $"Verification failed: {ex.Message}" };
            }
        }
    }
}
