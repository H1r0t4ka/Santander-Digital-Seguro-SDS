import { useState } from "react";
import { Header } from "~/components/header";
import FloatingChatbotButton from "~/components/FloatingChatbotButton";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Textarea } from "~/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Label } from "~/components/ui/label";

const ContactPage = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    phone: "",
    subject: "",
    message: "",
  });

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Form submitted:", formData);
    // Here you would typically send the data to your backend
    alert("Thank you for your message! We will contact you soon.");
    setFormData({
      name: "",
      email: "",
      phone: "",
      subject: "",
      message: "",
    });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="mx-auto max-w-2xl">
          <Card className="shadow-lg">
            <CardHeader className="text-center">
              <CardTitle className="text-3xl font-bold text-gray-900">
                Contáctenos
              </CardTitle>
              <p className="mt-2 text-muted-foreground">
                Estamos aquí para ayudarte. Envíanos un mensaje y te
                responderemos lo antes posible.
              </p>
            </CardHeader>

            <CardContent className="pt-6">
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="name">Nombre Completo</Label>
                    <Input
                      id="name"
                      name="name"
                      type="text"
                      placeholder="Ingresa tu nombre completo"
                      value={formData.name}
                      onChange={handleInputChange}
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="email">Correo Electrónico</Label>
                    <Input
                      id="email"
                      name="email"
                      type="email"
                      placeholder="Ingresa tu correo electrónico"
                      value={formData.email}
                      onChange={handleInputChange}
                      required
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="phone">Número de Teléfono</Label>
                    <Input
                      id="phone"
                      name="phone"
                      type="tel"
                      placeholder="Ingresa tu número de teléfono"
                      value={formData.phone}
                      onChange={handleInputChange}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="subject">Asunto</Label>
                    <Input
                      id="subject"
                      name="subject"
                      type="text"
                      placeholder="¿Sobre qué se trata?"
                      value={formData.subject}
                      onChange={handleInputChange}
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="message">Mensaje</Label>
                  <Textarea
                    id="message"
                    name="message"
                    placeholder="Por favor describe tu consulta o reporte en detalle..."
                    rows={6}
                    value={formData.message}
                    onChange={handleInputChange}
                    required
                    className="min-h-[150px]"
                  />
                </div>

                <Button type="submit" className="w-full" size="lg">
                  Enviar Mensaje
                </Button>
              </form>
            </CardContent>
          </Card>

          <div className="mt-8 text-center text-muted-foreground">
            <p className="mb-4 font-medium text-gray-900">
              También puedes contactarnos en:
            </p>

            <div className="space-y-2 text-sm">
              <p>
                <span className="font-semibold">Dirección:</span> Calle 37 No.
                10-30 Bucaramanga, Santander, Colombia.
              </p>
              <p>
                <span className="font-semibold">Código Postal:</span> 680006
              </p>
              <p>
                <span className="font-semibold">Horario de atención:</span>{" "}
                Lunes a Viernes de 7:30 a.m. a 12:00 m. y 1:00 p.m. a 5:00 p.m.
              </p>
              <p>
                <span className="font-semibold">
                  Tel (PBX) Palacio Amarillo:
                </span>{" "}
                (607) 6985868
              </p>
              <p>
                <span className="font-semibold">Correo electrónico PQRSD:</span>{" "}
                info@santander.gov.co
              </p>
            </div>
          </div>
        </div>
        <FloatingChatbotButton />
      </main>
    </div>
  );
};

export default ContactPage;
