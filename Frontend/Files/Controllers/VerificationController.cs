using Face_Recognition.Services;
using Microsoft.AspNetCore.Mvc;

namespace Face_Recognition.Controllers
{
    public class VerificationController : Controller
    {
        private readonly FaceVerificationService _faceService;
        private readonly IHttpClientFactory      _httpClientFactory;
        private readonly IConfiguration          _config;
        private readonly ILogger<VerificationController> _logger;

        public VerificationController(
            FaceVerificationService faceService,
            IHttpClientFactory      httpClientFactory,
            IConfiguration          config,
            ILogger<VerificationController> logger)
        {
            _faceService       = faceService;
            _httpClientFactory = httpClientFactory;
            _config            = config;
            _logger            = logger;
        }

        // GET /Verification/Index 
        public IActionResult Index() => View();

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ValidateId(IFormFile idImage)
        {
            if (idImage == null || idImage.Length == 0)
                return Json(new { valid = false, message = "Please upload an ID card image." });

            var result = await _faceService.ValidateIdAsync(idImage);
            return Json(new
            {
                valid    = result.Valid,
                idType   = result.IdType,
                message  = result.Message ?? result.Error
            });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Verify(
            IFormFile idImage,
            string    sessionId,
            string?   userName    = null,
            string?   userNameBn  = null,
            string?   userFather  = null,
            string?   userMother  = null,
            string?   userNid     = null,
            string?   userDob     = null)
        {
            _logger.LogInformation(
                "Verify called | sessionId={SessionId} | idImage={FileName} | size={Size}",
                sessionId ?? "(null)",
                idImage?.FileName ?? "(none)",
                idImage?.Length ?? 0);

            if (idImage == null || idImage.Length == 0)
            {
                _logger.LogWarning("Verify: no ID image uploaded");
                return View("Result", new FaceVerificationResult
                {
                    Error = "No ID card image was uploaded. Please go back and upload your ID card."
                });
            }

            if (string.IsNullOrWhiteSpace(sessionId))
            {
                _logger.LogWarning("Verify: empty sessionId — liveness JS did not populate hidden field");
                return View("Result", new FaceVerificationResult
                {
                    Error = "Liveness session is missing. This usually means the liveness check " +
                            "did not complete successfully. Please go back and complete all 3 " +
                            "challenges (blink, head turn, motion) before clicking Verify."
                });
            }

            FaceVerificationResult model;
            try
            {
                _logger.LogInformation("Calling Python /verify-with-liveness | session={SessionId}", sessionId);
                model = await _faceService.VerifyWithLivenessAsync(idImage, sessionId);
                _logger.LogInformation(
                    "API returned | match={Match} | similarity={Sim:F4} | error={Error}",
                    model.Match, model.Similarity, model.Error ?? "none");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Unexpected exception calling VerifyWithLivenessAsync");
                model = new FaceVerificationResult
                {
                    Error = $"Could not reach the verification service: {ex.Message}. " +
                             "Please ensure both Docker containers are running."
                };
            }

            model.UserName   = userName;
            model.UserNameBn = userNameBn;
            model.UserFather = userFather;
            model.UserMother = userMother;
            model.UserNid    = userNid;
            model.UserDob    = userDob;

            if (model.Error == null && model.NidData != null)
            {
                static string Norm(string? s) =>
                    System.Text.RegularExpressions.Regex
                        .Replace((s ?? "").Trim(), @"\s+", " ")
                        .ToLowerInvariant();

                if (!string.IsNullOrWhiteSpace(userName))
                {
                    var ocr = string.IsNullOrWhiteSpace(model.NidData.NameEn)
                        ? model.NidData.Name : model.NidData.NameEn;
                    model.NameMatch = string.Equals(Norm(userName), Norm(ocr),
                        StringComparison.OrdinalIgnoreCase);
                }

                if (!string.IsNullOrWhiteSpace(userNameBn))
                    model.NameBnMatch = string.Equals(Norm(userNameBn), Norm(model.NidData.NameBn),
                        StringComparison.OrdinalIgnoreCase);

                if (!string.IsNullOrWhiteSpace(userFather))
                    model.FatherMatch = string.Equals(Norm(userFather), Norm(model.NidData.FatherBn),
                        StringComparison.OrdinalIgnoreCase);

                if (!string.IsNullOrWhiteSpace(userMother))
                    model.MotherMatch = string.Equals(Norm(userMother), Norm(model.NidData.MotherBn),
                        StringComparison.OrdinalIgnoreCase);

                if (!string.IsNullOrWhiteSpace(userNid))
                {
                    string cleanInput = System.Text.RegularExpressions.Regex.Replace(userNid, @"[\s\-\.]", "");
                    string cleanOcr   = System.Text.RegularExpressions.Regex.Replace(
                        model.NidData.NidNumber, @"[\s\-\.]", "");
                    model.NidMatch = string.Equals(cleanInput, cleanOcr, StringComparison.OrdinalIgnoreCase);
                }

                // Date of birth
                if (!string.IsNullOrWhiteSpace(userDob))
                    model.DobMatch = string.Equals(Norm(userDob), Norm(model.NidData.Dob),
                        StringComparison.OrdinalIgnoreCase);
            }

            return View("Result", model);
        }

        private string PythonBase =>
            _config["FaceApiBaseUrl"] ?? "http://localhost:8000";

        // POST /Verification/LivenessStart
        [HttpPost]
        public async Task<IActionResult> LivenessStart()
        {
            try
            {
                var client   = _httpClientFactory.CreateClient("PythonApi");
                var response = await client.PostAsync("/liveness/start", new StringContent(""));
                var json     = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogError("LivenessStart: Python returned {Status}: {Body}",
                        (int)response.StatusCode, json);
                    return StatusCode((int)response.StatusCode, json);
                }

                return Content(json, "application/json");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "LivenessStart: Python API unreachable");
                return StatusCode(503, new { error = $"Python API unavailable: {ex.Message}" });
            }
        }

        // POST /Verification/LivenessFrame?sessionId=xxx 
        [HttpPost]
        public async Task<IActionResult> LivenessFrame(
            [FromQuery] string    sessionId,
            [FromForm]  IFormFile frame)
        {
            if (string.IsNullOrEmpty(sessionId) || frame == null || frame.Length == 0)
                return BadRequest(new { error = "Missing sessionId or frame" });

            try
            {
                var client = _httpClientFactory.CreateClient("PythonApi");

                using var frameStream = new MemoryStream();
                await frame.CopyToAsync(frameStream);

                using var form      = new MultipartFormDataContent();
                var frameBytes      = new ByteArrayContent(frameStream.ToArray());
                frameBytes.Headers.ContentType =
                    new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");
                form.Add(frameBytes, "frame", "frame.jpg");

                var response = await client.PostAsync(
                    $"/liveness/frame?session_id={Uri.EscapeDataString(sessionId)}",
                    form);

                var json = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                    return StatusCode((int)response.StatusCode, json);

                return Content(json, "application/json");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "LivenessFrame: Python API unreachable");
                return StatusCode(503, new { error = $"Python API unavailable: {ex.Message}" });
            }
        }

        // POST /Verification/LivenessCancel?sessionId=xxx
        [HttpPost]
        public async Task<IActionResult> LivenessCancel([FromQuery] string sessionId)
        {
            if (string.IsNullOrEmpty(sessionId))
                return Ok(new { cancelled = true });

            try
            {
                var client = _httpClientFactory.CreateClient("PythonApi");
                await client.PostAsync(
                    $"/liveness/cancel?session_id={Uri.EscapeDataString(sessionId)}",
                    new StringContent(""));
            }
            catch
            {
                // Cancel errors are non-critical — ignore
            }

            return Ok(new { cancelled = true });
        }
    }
}
