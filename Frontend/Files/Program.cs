using Face_Recognition.Services;
using Microsoft.AspNetCore.Http.Features;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllersWithViews();
builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit    = 52_428_800; // 50 MB
    options.ValueLengthLimit            = 52_428_800;
    options.MultipartHeadersLengthLimit = 52_428_800;
});

builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = 52_428_800; // 50 MB
});

var apiUrl = builder.Configuration["FaceApiBaseUrl"] ?? "http://localhost:8000";

builder.Services.AddHttpClient<FaceVerificationService>(client =>
{
    client.BaseAddress = new Uri(apiUrl);
    client.Timeout     = TimeSpan.FromSeconds(90);
});

builder.Services.AddHttpClient("PythonApi", client =>
{
    client.BaseAddress = new Uri(apiUrl);
    client.Timeout     = TimeSpan.FromSeconds(30);
});

builder.Logging.ClearProviders();
builder.Logging.AddConsole();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseStaticFiles();
app.UseRouting();
app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Verification}/{action=Index}/{id?}");

app.Run();
