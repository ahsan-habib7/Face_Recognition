using System.Net.Http.Headers;
using System.Text.Json;

namespace Face_Recognition.Services
{
    public class IdValidationResult
    {
        public bool    Valid   { get; set; }
        public string  IdType  { get; set; } = string.Empty;
        public string  Message { get; set; } = string.Empty;
        public string? Error   { get; set; }
    }

    public class FaceVerificationResult
    {
        public bool    Match      { get; set; }
        public string  Message    { get; set; } = string.Empty;
        public double  Similarity { get; set; }
        public string  Confidence { get; set; } = string.Empty;
        public double  Threshold  { get; set; }
        public string? Error      { get; set; }
    }

    // ── Service ────────────────────────────────────────────────
    public class FaceVerificationService
    {
        private readonly HttpClient _httpClient;

        public FaceVerificationService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public async Task<IdValidationResult> ValidateIdAsync(IFormFile idImage)
        {
            try
            {
                using var idStream = new MemoryStream();
                await idImage.CopyToAsync(idStream);

                using var form      = new MultipartFormDataContent();
                var       idContent = new ByteArrayContent(idStream.ToArray());
                idContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
                form.Add(idContent, "id_image", "id_image.jpg");

                var response = await _httpClient.PostAsync("/validate-id", form);
                var json     = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    using var errDoc = JsonDocument.Parse(json);
                    var detail = errDoc.RootElement
                                       .TryGetProperty("detail", out var d)
                                       ? d.GetString()
                                       : "ID validation failed.";
                    return new IdValidationResult { Valid = false, Error = detail };
                }

                var result = JsonSerializer.Deserialize<IdValidationResult>(
                    json,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true }
                );

                return result ?? new IdValidationResult
                {
                    Valid = false,
                    Error = "Empty response from validation API."
                };
            }
            catch (HttpRequestException ex)
            {
                return new IdValidationResult
                {
                    Valid  = false,
                    Error  = $"Cannot connect to API. Make sure Docker is running. {ex.Message}"
                };
            }
            catch (Exception ex)
            {
                return new IdValidationResult
                {
                    Valid = false,
                    Error = $"Unexpected error: {ex.Message}"
                };
            }
        }

        public async Task<FaceVerificationResult> VerifyAsync(
            IFormFile idImage,
            IFormFile selfieImage)
        {
            try
            {
                using var idStream     = new MemoryStream();
                using var selfieStream = new MemoryStream();

                await idImage.CopyToAsync(idStream);
                await selfieImage.CopyToAsync(selfieStream);

                using var form        = new MultipartFormDataContent();
                var       idContent   = new ByteArrayContent(idStream.ToArray());
                var       selfContent = new ByteArrayContent(selfieStream.ToArray());

                idContent.Headers.ContentType   = new MediaTypeHeaderValue("image/jpeg");
                selfContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");

                form.Add(idContent,   "id_image",     "id_image.jpg");
                form.Add(selfContent, "selfie_image", "selfie_image.jpg");

                var response = await _httpClient.PostAsync("/verify-face", form);
                var json     = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    using var errDoc = JsonDocument.Parse(json);
                    var detail = errDoc.RootElement
                                       .TryGetProperty("detail", out var d)
                                       ? d.GetString()
                                       : "Face verification failed.";
                    return new FaceVerificationResult { Error = detail };
                }

                var result = JsonSerializer.Deserialize<FaceVerificationResult>(
                    json,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true }
                );

                return result ?? new FaceVerificationResult
                {
                    Error = "Empty response from face verification API."
                };
            }
            catch (HttpRequestException ex)
            {
                return new FaceVerificationResult
                {
                    Error = $"Cannot connect to API. Make sure Docker is running. {ex.Message}"
                };
            }
            catch (TaskCanceledException)
            {
                return new FaceVerificationResult
                {
                    Error = "Request timed out. Please try again."
                };
            }
            catch (Exception ex)
            {
                return new FaceVerificationResult
                {
                    Error = $"Unexpected error: {ex.Message}"
                };
            }
        }
    }
}
